# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import sys
import base64
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import argparse
from omegaconf import OmegaConf

import torch
import zipfile
from glob import glob
import moviepy.editor as mpy
from lam.utils.ffmpeg_utils import images_to_video
from tools.flame_tracking_single_image import FlameTrackingSingleImage
from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


app = FastAPI(title="LAM Image to ZIP API", description="Convert images to animatable avatars")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
lam = None
flametracking = None
cfg = None


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--infer", type=str)
    parser.add_argument("--blender_path", type=str,
                        default='blender', help="Path to Blender executable")
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    cfg.blender_path = args.blender_path

    # parse from ENV
    if os.environ.get("APP_INFER") is not None:
        args.infer = os.environ.get("APP_INFER")
    if os.environ.get("APP_MODEL_NAME") is not None:
        cli_cfg.model_name = os.environ.get("APP_MODEL_NAME")

    args.config = args.infer if args.config is None else args.config

    if args.config is not None:
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault(
            "save_tmp_dump", os.path.join("exps", cli_cfg.model_name, "save_tmp")
        )
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, "images"))
        cfg.setdefault(
            "video_dump", os.path.join("dumps", cli_cfg.model_name, "videos")
        )
        cfg.setdefault("mesh_dump", os.path.join("dumps", cli_cfg.model_name, "meshes"))

    cfg.motion_video_read_fps = 30
    cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"

    return cfg, None


def _build_model(cfg):
    from lam.models import ModelLAM
    from safetensors.torch import load_file

    model = ModelLAM(**cfg.model)
    resume = os.path.join(cfg.model_name, "model.safetensors")
    print("="*100)
    print("loading pretrained weight from:", resume)
    if resume.endswith('safetensors'):
        ckpt = load_file(resume, device='cpu')
    else:
        ckpt = torch.load(resume, map_location='cpu')
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f"WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.")
        else:
            print(f"WARN] unexpected param {k}: {v.shape}")
    print("finish loading pretrained weight from:", resume)
    print("="*100)
    return model


def add_audio_to_video(video_path, out_path, audio_path):
    from moviepy.editor import VideoFileClip, AudioFileClip
    
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    video_clip_with_audio = video_clip.set_audio(audio_clip)
    video_clip_with_audio.write_videofile(out_path, codec='libx264', audio_codec='aac')
    print(f"Audio added successfully at {out_path}")


def save_images2video(img_lst, v_pth, fps):
    from moviepy.editor import ImageSequenceClip
    images = [image.astype(np.uint8) for image in img_lst]
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(v_pth, codec='libx264')
    print(f"Video saved successfully at {v_pth}")


def process_image_to_zip(image_path: str, motion_name: str = "nice") -> tuple[str, str, str]:
    """
    Process image and create ZIP file for Open Avatar Chat
    Returns: (processed_image_path, video_path, zip_path)
    """
    global lam, flametracking, cfg
    
    with tempfile.TemporaryDirectory() as working_dir:
        image_raw = os.path.join(working_dir, "raw.png")
        
        # Copy uploaded image to working directory
        shutil.copy2(image_path, image_raw)
        
        base_vid = motion_name
        flame_params_dir = os.path.join("./assets/sample_motion/export", base_vid, "flame_param")
        
        if not os.path.exists(flame_params_dir):
            raise HTTPException(status_code=400, detail=f"Motion '{motion_name}' not found")
        
        base_iid = "uploaded_image"
        dump_video_path = os.path.join(working_dir, "output.mp4")
        dump_image_path = os.path.join(working_dir, "output.png")

        motion_seqs_dir = flame_params_dir
        dump_image_dir = os.path.dirname(dump_image_path)
        os.makedirs(dump_image_dir, exist_ok=True)
        dump_tmp_dir = dump_image_dir

        motion_img_need_mask = cfg.get("motion_img_need_mask", False)
        vis_motion = cfg.get("vis_motion", False)

        # Preprocess input image: segmentation, flame params estimation
        return_code = flametracking.preprocess(image_raw)
        if return_code != 0:
            raise HTTPException(status_code=500, detail="Flame tracking preprocess failed")
            
        return_code = flametracking.optimize()
        if return_code != 0:
            raise HTTPException(status_code=500, detail="Flame tracking optimize failed")
            
        return_code, output_dir = flametracking.export()
        if return_code != 0:
            raise HTTPException(status_code=500, detail="Flame tracking export failed")

        image_path = os.path.join(output_dir, "images/00000_00.png")
        mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")

        aspect_standard = 1.0/1.0
        source_size = cfg.source_size
        render_size = cfg.render_size
        render_fps = 30
        
        # Prepare reference image
        image, _, _, shape_param = preprocess_image(
            image_path, mask_path=mask_path, intr=None, pad_ratio=0, bg_color=1., 
            max_tgt_size=None, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size, multiply=14, need_mask=True, get_shape_param=True
        )

        # Save masked image for visualization
        save_ref_img_path = os.path.join(dump_tmp_dir, "output.png")
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # Prepare motion sequence
        src = "uploaded_image"
        driven = motion_seqs_dir.split('/')[-2]
        src_driven = [src, driven]
        motion_seq = prepare_motion_seqs(
            motion_seqs_dir, None, save_root=dump_tmp_dir, fps=render_fps,
            bg_color=1., aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
            render_image_res=render_size, multiply=16, 
            need_mask=motion_img_need_mask, vis_motion=vis_motion, 
            shape_param=shape_param, test_sample=False, cross_id=False, src_driven=src_driven
        )

        # Start inference
        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        device, dtype = "cuda", torch.float32
        
        with torch.no_grad():
            res = lam.infer_single_view(
                image.unsqueeze(0).to(device, dtype), None, None, 
                render_c2ws=motion_seq["render_c2ws"].to(device),
                render_intrs=motion_seq["render_intrs"].to(device),
                render_bg_colors=motion_seq["render_bg_colors"].to(device),
                flame_params={k: v.to(device) for k, v in motion_seq["flame_params"].items()}
            )

        # Create ZIP file for Open Avatar Chat
        try:
            from tools.generateARKITGLBWithBlender import generate_glb
            import patoolib

            oac_dir = os.path.join(working_dir, 'open_avatar_chat', base_iid)
            os.makedirs(oac_dir, exist_ok=True)
            
            saved_head_path = lam.renderer.flame_model.save_shaped_mesh(shape_param.unsqueeze(0).cuda(), fd=oac_dir)
            res['cano_gs_lst'][0].save_ply(os.path.join(oac_dir, "offset.ply"), rgb2sh=False, offset2xyz=True)
            
            generate_glb(
                input_mesh=Path(saved_head_path),
                template_fbx=Path("./assets/sample_oac/template_file.fbx"),
                output_glb=Path(os.path.join(oac_dir, "skin.glb")),
                blender_exec=Path(cfg.blender_path)
            )
            
            shutil.copy(
                src='./assets/sample_oac/animation.glb',
                dst=os.path.join(oac_dir, 'animation.glb')
            )
            os.remove(saved_head_path)

            # Create final ZIP file in a permanent location
            output_zip_path = os.path.join('./output/api_exports', f'{base_iid}_{motion_name}.zip')
            os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)
            
            if os.path.exists(output_zip_path):
                os.remove(output_zip_path)
                
            original_cwd = os.getcwd()
            oac_parent_dir = os.path.dirname(oac_dir)
            base_iid_dir = os.path.basename(oac_dir)
            os.chdir(oac_parent_dir)
            try:
                patoolib.create_archive(
                    archive=os.path.abspath(output_zip_path),
                    filenames=[base_iid_dir],
                    verbosity=-1,
                    program='zip'
                )
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Archive creation failed: {str(e)}")

        # Generate video
        rgb = res["comp_rgb"].detach().cpu().numpy()
        mask = res["comp_mask"].detach().cpu().numpy()
        mask[mask < 0.5] = 0.0
        rgb = rgb * mask + (1 - mask) * 1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)

        # Save video
        final_video_path = os.path.join('./output/api_exports', f'{base_iid}_{motion_name}.mp4')
        os.makedirs(os.path.dirname(final_video_path), exist_ok=True)
        save_images2video(rgb, final_video_path, render_fps)
        
        # Add audio if available
        audio_path = os.path.join("./assets/sample_motion/export", base_vid, base_vid+".wav")
        if os.path.exists(audio_path):
            final_video_path_wa = final_video_path.replace(".mp4", "_audio.mp4")
            add_audio_to_video(final_video_path, final_video_path_wa, audio_path)
            final_video_path = final_video_path_wa

        # Copy processed image to permanent location
        final_image_path = os.path.join('./output/api_exports', f'{base_iid}_{motion_name}_processed.png')
        shutil.copy2(save_ref_img_path, final_image_path)
        
        return final_image_path, final_video_path, output_zip_path


def initialize_models():
    """Initialize models on startup"""
    global lam, flametracking, cfg
    
    # Set environment variables
    os.environ.update({
        'APP_ENABLED': '1',
        'APP_MODEL_NAME': './model_zoo/lam_models/releases/lam/lam-20k/step_045500/',
        'APP_INFER': './configs/inference/lam-20k-8gpu.yaml',
        'APP_TYPE': 'infer.lam',
        'NUMBA_THREADING_LAYER': 'omp',
    })

    # Parse configs and build model
    cfg, _ = parse_configs()
    lam = _build_model(cfg)
    lam.to('cuda')
    lam.eval()

    # Initialize flame tracking
    flametracking = FlameTrackingSingleImage(
        output_dir='output/tracking',
        alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
        vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
        human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
        facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
        detect_iris_landmarks=False
    )
    
    print("Models initialized successfully!")


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    initialize_models()


@app.get("/")
async def root():
    return {"message": "LAM Image to ZIP API", "status": "ready"}


@app.get("/motions")
async def list_available_motions():
    """List all available motion sequences"""
    motion_dir = "./assets/sample_motion/export"
    if not os.path.exists(motion_dir):
        return {"motions": []}
    
    motions = []
    for item in os.listdir(motion_dir):
        motion_path = os.path.join(motion_dir, item)
        if os.path.isdir(motion_path):
            flame_param_dir = os.path.join(motion_path, "flame_param")
            if os.path.exists(flame_param_dir):
                motions.append(item)
    
    return {"motions": motions}


@app.post("/process")
async def process_image(
    image: UploadFile = File(..., description="Input image file"),
    motion: str = Form("nice", description="Motion sequence name")
):
    """
    Process an uploaded image and create ZIP file for Open Avatar Chat
    """
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        contents = await image.read()
        tmp_file.write(contents)
        tmp_image_path = tmp_file.name
    
    try:
        # Process the image
        processed_image_path, video_path, zip_path = process_image_to_zip(tmp_image_path, motion)
        
        return {
            "status": "success",
            "message": "Image processed successfully",
            "results": {
                "processed_image": processed_image_path,
                "video": video_path,
                "zip_file": zip_path
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_image_path):
            os.unlink(tmp_image_path)


@app.get("/download/{file_type}/{filename}")
async def download_file(file_type: str, filename: str):
    """
    Download generated files (image, video, or zip)
    """
    valid_types = ["image", "video", "zip"]
    if file_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Must be one of: {valid_types}")
    
    file_path = os.path.join('./output/api_exports', filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type based on file extension
    if filename.endswith('.zip'):
        media_type = 'application/zip'
    elif filename.endswith('.mp4'):
        media_type = 'video/mp4'
    elif filename.endswith('.png'):
        media_type = 'image/png'
    elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
        media_type = 'image/jpeg'
    else:
        media_type = 'application/octet-stream'
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 