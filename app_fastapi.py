#!/usr/bin/env python3
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
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List
import asyncio
import logging

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Core processing imports
import cv2
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf
from glob import glob

# LAM specific imports
from tools.flame_tracking_single_image import FlameTrackingSingleImage
from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image
from lam.utils.ffmpeg_utils import images_to_video

# Import utility functions from the original app
from app_lam import (
    parse_configs, _build_model, save_images2video, add_audio_to_video,
    create_zip_archive, h5_rendering
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
lam_model = None
flametracking_model = None
cfg = None

# Pydantic models for API
class ProcessRequest(BaseModel):
    motion_video: str = "nice"  # Default motion video name
    enable_oac_file: bool = True  # Enable ZIP file generation
    
class ProcessResponse(BaseModel):
    success: bool
    message: str
    processed_image_url: Optional[str] = None
    output_video_url: Optional[str] = None
    zip_file_url: Optional[str] = None
    job_id: str

class StatusResponse(BaseModel):
    status: str
    message: str
    available_motions: List[str]

# FastAPI app
app = FastAPI(
    title="LAM API - Large Avatar Model",
    description="API for processing images with LAM (Large Avatar Model) to generate animatable Gaussian heads",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def setup_environment():
    """Setup environment variables for LAM"""
    os.environ.update({
        'APP_ENABLED': '1',
        'APP_MODEL_NAME': './model_zoo/lam_models/releases/lam/lam-20k/step_045500/',
        'APP_INFER': './configs/inference/lam-20k-8gpu.yaml',
        'APP_TYPE': 'infer.lam',
        'NUMBA_THREADING_LAYER': 'omp',
    })

def get_available_motions():
    """Get list of available motion videos"""
    motion_path = "./assets/sample_motion/export"
    if not os.path.exists(motion_path):
        return []
    
    motions = []
    for motion_dir in os.listdir(motion_path):
        motion_full_path = os.path.join(motion_path, motion_dir)
        if os.path.isdir(motion_full_path):
            # Check if required files exist
            flame_params_dir = os.path.join(motion_full_path, "flame_param")
            audio_file = os.path.join(motion_full_path, f"{motion_dir}.wav")
            if os.path.exists(flame_params_dir):
                motions.append(motion_dir)
    
    return motions

async def process_image_core(image_path: str, motion_video: str, enable_oac_file: bool, working_dir: str):
    """Core image processing function adapted from the original Gradio app"""
    try:
        base_vid = motion_video
        flame_params_dir = os.path.join("./assets/sample_motion/export", base_vid, "flame_param")
        
        # Validate motion exists
        if not os.path.exists(flame_params_dir):
            raise ValueError(f"Motion '{motion_video}' not found. Available motions: {get_available_motions()}")
        
        dump_video_path = os.path.join(working_dir, "output.mp4")
        dump_image_path = os.path.join(working_dir, "output.png")
        dump_video_path_wa = dump_video_path.replace(".mp4", "_audio.mp4")
        
        # Create output directories
        os.makedirs(os.path.dirname(dump_image_path), exist_ok=True)
        
        motion_img_need_mask = cfg.get("motion_img_need_mask", False)
        vis_motion = cfg.get("vis_motion", False)
        
        # Preprocess input image: segmentation, flame params estimation
        logger.info("Starting flame tracking preprocessing...")
        return_code = flametracking_model.preprocess(image_path)
        if return_code != 0:
            raise RuntimeError("Flametracking preprocess failed!")
        
        return_code = flametracking_model.optimize()
        if return_code != 0:
            raise RuntimeError("Flametracking optimize failed!")
        
        return_code, output_dir = flametracking_model.export()
        if return_code != 0:
            raise RuntimeError("Flametracking export failed!")
        
        processed_image_path = os.path.join(output_dir, "images/00000_00.png")
        mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")
        
        logger.info(f"Processed image: {processed_image_path}")
        logger.info(f"Mask path: {mask_path}")
        
        # Prepare reference image
        aspect_standard = 1.0
        source_size = cfg.source_size
        render_size = cfg.render_size
        render_fps = 30
        
        image, _, _, shape_param = preprocess_image(
            processed_image_path, mask_path=mask_path, intr=None, pad_ratio=0, 
            bg_color=1., max_tgt_size=None, aspect_standard=aspect_standard, 
            enlarge_ratio=[1.0, 1.0], render_tgt_size=source_size, multiply=14, 
            need_mask=True, get_shape_param=True
        )
        
        # Save masked image for visualization
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(dump_image_path)
        
        # Prepare motion sequence
        src = os.path.basename(image_path).split('.')[0]
        driven = motion_video
        src_driven = [src, driven]
        
        logger.info("Preparing motion sequences...")
        motion_seq = prepare_motion_seqs(
            flame_params_dir, None, save_root=working_dir, fps=render_fps,
            bg_color=1., aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
            render_image_res=render_size, multiply=16, need_mask=motion_img_need_mask,
            vis_motion=vis_motion, shape_param=shape_param, test_sample=False,
            cross_id=False, src_driven=src_driven
        )
        
        # Start inference
        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        device, dtype = "cuda", torch.float32
        
        logger.info("Starting LAM inference...")
        with torch.no_grad():
            res = lam_model.infer_single_view(
                image.unsqueeze(0).to(device, dtype), None, None,
                render_c2ws=motion_seq["render_c2ws"].to(device),
                render_intrs=motion_seq["render_intrs"].to(device),
                render_bg_colors=motion_seq["render_bg_colors"].to(device),
                flame_params={k: v.to(device) for k, v in motion_seq["flame_params"].items()}
            )
        
        # Handle ZIP file creation for Open Avatar Chat
        output_zip_path = ""
        if enable_oac_file:
            try:
                logger.info("Creating OAC ZIP file...")
                from tools.generateARKITGLBWithBlender import generate_glb
                import zipfile
                
                base_iid = os.path.basename(image_path).split('.')[0]
                oac_dir = os.path.join(working_dir, 'open_avatar_chat', base_iid)
                os.makedirs(oac_dir, exist_ok=True)
                
                saved_head_path = lam_model.renderer.flame_model.save_shaped_mesh(
                    shape_param.unsqueeze(0).cuda(), fd=oac_dir
                )
                res['cano_gs_lst'][0].save_ply(
                    os.path.join(oac_dir, "offset.ply"), rgb2sh=False, offset2xyz=True
                )
                
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
                
                output_zip_path = os.path.join(working_dir, base_iid + '_oac.zip')
                if os.path.exists(output_zip_path):
                    os.remove(output_zip_path)
                
                # Create ZIP file using built-in zipfile module
                with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(oac_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arc_name = os.path.relpath(file_path, os.path.dirname(oac_dir))
                            zipf.write(file_path, arc_name)
                
                shutil.rmtree(oac_dir)
                logger.info(f"OAC ZIP file created: {output_zip_path}")
                
            except Exception as e:
                logger.error(f"OAC ZIP creation failed: {str(e)}")
                output_zip_path = f"Archive creation failed: {str(e)}"
        
        # Generate video
        logger.info("Generating output video...")
        rgb = res["comp_rgb"].detach().cpu().numpy()
        mask = res["comp_mask"].detach().cpu().numpy()
        mask[mask < 0.5] = 0.0
        rgb = rgb * mask + (1 - mask) * 1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        
        if vis_motion:
            vis_ref_img_resized = np.tile(
                cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]), 
                          interpolation=cv2.INTER_AREA)[None, :, :, :],
                (rgb.shape[0], 1, 1, 1),
            )
            rgb = np.concatenate([vis_ref_img_resized, rgb, motion_seq["vis_motion_render"]], axis=2)
        
        save_images2video(rgb, dump_video_path, render_fps)
        
        # Add audio if available
        audio_path = os.path.join("./assets/sample_motion/export", base_vid, base_vid + ".wav")
        if os.path.exists(audio_path):
            add_audio_to_video(dump_video_path, dump_video_path_wa, audio_path)
        else:
            dump_video_path_wa = dump_video_path
            logger.warning(f"Audio file not found: {audio_path}")
        
        return dump_image_path, dump_video_path_wa, output_zip_path
        
    except Exception as e:
        logger.error(f"Error in core processing: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global lam_model, flametracking_model, cfg
    
    logger.info("Starting LAM API server...")
    
    # Setup environment
    setup_environment()
    
    # Parse configuration
    cfg, _ = parse_configs()
    
    # Initialize LAM model
    logger.info("Loading LAM model...")
    lam_model = _build_model(cfg)
    lam_model.to('cuda')
    lam_model.eval()
    
    # Initialize FlameTracking model
    logger.info("Loading FlameTracking model...")
    flametracking_model = FlameTrackingSingleImage(
        output_dir='output/tracking',
        alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
        vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
        human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
        facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
        detect_iris_landmarks=False
    )
    
    logger.info("LAM API server startup complete!")

@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint with API status"""
    return StatusResponse(
        status="running",
        message="LAM API is running successfully",
        available_motions=get_available_motions()
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": lam_model is not None and flametracking_model is not None}

@app.get("/motions")
async def list_motions():
    """List available motion videos"""
    return {"motions": get_available_motions()}

@app.post("/process", response_model=ProcessResponse)
async def process_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    motion_video: str = "nice",
    enable_oac_file: bool = True
):
    """
    Process an uploaded image with LAM model
    
    - **image**: Input image file (JPEG, PNG)
    - **motion_video**: Name of motion video to use (default: "nice")
    - **enable_oac_file**: Whether to generate ZIP file for Open Avatar Chat (default: True)
    """
    
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate motion exists
    available_motions = get_available_motions()
    if motion_video not in available_motions:
        raise HTTPException(
            status_code=400, 
            detail=f"Motion '{motion_video}' not found. Available: {available_motions}"
        )
    
    # Create temporary working directory
    working_dir = tempfile.mkdtemp(prefix="lam_api_")
    job_id = os.path.basename(working_dir)
    
    try:
        # Save uploaded image
        image_path = os.path.join(working_dir, "input.png")
        with open(image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        logger.info(f"Processing job {job_id} with motion '{motion_video}'")
        
        # Process image
        processed_image_path, output_video_path, zip_file_path = await process_image_core(
            image_path, motion_video, enable_oac_file, working_dir
        )
        
        # Prepare response URLs (relative to working directory for serving)
        response = ProcessResponse(
            success=True,
            message="Image processed successfully",
            processed_image_url=f"/download/{job_id}/processed_image.png",
            output_video_url=f"/download/{job_id}/output_video.mp4",
            zip_file_url=f"/download/{job_id}/output.zip" if enable_oac_file and zip_file_path and not zip_file_path.startswith("Archive creation failed") else None,
            job_id=job_id
        )
        
        # Copy files to accessible locations
        final_processed_path = os.path.join(working_dir, "processed_image.png")
        final_video_path = os.path.join(working_dir, "output_video.mp4")
        final_zip_path = os.path.join(working_dir, "output.zip")
        
        shutil.copy(processed_image_path, final_processed_path)
        shutil.copy(output_video_path, final_video_path)
        
        if enable_oac_file and zip_file_path and not zip_file_path.startswith("Archive creation failed"):
            shutil.copy(zip_file_path, final_zip_path)
        
        # Schedule cleanup after 1 hour
        background_tasks.add_task(cleanup_working_dir, working_dir, delay=3600)
        
        return response
        
    except Exception as e:
        # Cleanup on error
        shutil.rmtree(working_dir, ignore_errors=True)
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download processed files"""
    # Security: validate job_id and filename
    if not job_id.replace('_', '').replace('-', '').isalnum():
        raise HTTPException(status_code=400, detail="Invalid job ID")
    
    allowed_files = ["processed_image.png", "output_video.mp4", "output.zip"]
    if filename not in allowed_files:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = os.path.join("/tmp", f"lam_api_{job_id}", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

async def cleanup_working_dir(working_dir: str, delay: int = 0):
    """Clean up working directory after delay"""
    if delay > 0:
        await asyncio.sleep(delay)
    
    try:
        shutil.rmtree(working_dir, ignore_errors=True)
        logger.info(f"Cleaned up working directory: {working_dir}")
    except Exception as e:
        logger.error(f"Failed to cleanup {working_dir}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 