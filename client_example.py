#!/usr/bin/env python3
"""
LAM FastAPI Client Example

This script demonstrates how to interact with the LAM FastAPI service
to process images and generate ZIP files for avatar animations.
"""

import requests
import json
import time
import os
from pathlib import Path
import argparse


class LAMAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def health_check(self):
        """Check if the API is healthy and models are loaded"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Health check failed: {e}")
            return None
    
    def get_available_motions(self):
        """Get list of available motion videos"""
        try:
            response = self.session.get(f"{self.base_url}/motions")
            response.raise_for_status()
            return response.json()["motions"]
        except Exception as e:
            print(f"Failed to get motions: {e}")
            return []
    
    def get_status(self):
        """Get API status and available motions"""
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to get status: {e}")
            return None
    
    def process_image(self, image_path, motion_video="nice", enable_oac_file=True, timeout=300):
        """
        Process an image with the LAM model
        
        Args:
            image_path (str): Path to the input image
            motion_video (str): Name of the motion video to use
            enable_oac_file (bool): Whether to generate ZIP file for Open Avatar Chat
            timeout (int): Request timeout in seconds
            
        Returns:
            dict: Processing results or None if failed
        """
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
        
        try:
            # Prepare files and data
            with open(image_path, 'rb') as f:
                files = {'image': (os.path.basename(image_path), f, 'image/png')}
                data = {
                    'motion_video': motion_video,
                    'enable_oac_file': enable_oac_file
                }
                
                print(f"Processing image: {image_path}")
                print(f"Motion video: {motion_video}")
                print(f"Enable OAC file: {enable_oac_file}")
                print("Uploading and processing... This may take several minutes.")
                
                response = self.session.post(
                    f"{self.base_url}/process",
                    files=files,
                    data=data,
                    timeout=timeout
                )
                response.raise_for_status()
                return response.json()
                
        except requests.exceptions.Timeout:
            print(f"Request timed out after {timeout} seconds")
            return None
        except Exception as e:
            print(f"Processing failed: {e}")
            return None
    
    def download_file(self, job_id, filename, output_path):
        """
        Download a processed file
        
        Args:
            job_id (str): Job ID from processing response
            filename (str): Name of file to download
            output_path (str): Local path to save the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/download/{job_id}/{filename}",
                stream=True
            )
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded: {output_path}")
            return True
            
        except Exception as e:
            print(f"Download failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="LAM API Client Example")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--motion", default="nice", help="Motion video name (default: nice)")
    parser.add_argument("--output-dir", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--no-zip", action="store_true", help="Disable ZIP file generation")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    # Initialize client
    client = LAMAPIClient(args.base_url)
    
    # Check API health
    print("Checking API health...")
    health = client.health_check()
    if not health:
        print("API is not accessible!")
        return 1
    
    print(f"API Health: {health}")
    
    if not health.get("models_loaded", False):
        print("Models are not loaded!")
        return 1
    
    # Get available motions
    print("\nGetting available motions...")
    motions = client.get_available_motions()
    print(f"Available motions: {motions}")
    
    if args.motion not in motions:
        print(f"Motion '{args.motion}' not available!")
        print(f"Available motions: {motions}")
        return 1
    
    # Process image
    print(f"\nProcessing image with motion '{args.motion}'...")
    start_time = time.time()
    
    result = client.process_image(
        image_path=args.image,
        motion_video=args.motion,
        enable_oac_file=not args.no_zip,
        timeout=args.timeout
    )
    
    if not result:
        print("Processing failed!")
        return 1
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Download results
    if result.get("success"):
        job_id = result["job_id"]
        output_dir = Path(args.output_dir) / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download processed image
        if result.get("processed_image_url"):
            client.download_file(
                job_id, "processed_image.png", 
                str(output_dir / "processed_image.png")
            )
        
        # Download output video
        if result.get("output_video_url"):
            client.download_file(
                job_id, "output_video.mp4", 
                str(output_dir / "output_video.mp4")
            )
        
        # Download ZIP file
        if result.get("zip_file_url"):
            client.download_file(
                job_id, "output.zip", 
                str(output_dir / "avatar_chat.zip")
            )
        
        print(f"\nAll files downloaded to: {output_dir}")
        
    return 0


if __name__ == "__main__":
    exit(main()) 