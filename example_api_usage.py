#!/usr/bin/env python3
"""
Example script showing how to use the LAM FastAPI for image-to-zip conversion.
"""

import requests
import os
from pathlib import Path


def test_api(api_url="http://localhost:8000", image_path="assets/sample_input/james.png", motion="nice"):
    """
    Test the LAM FastAPI by uploading an image and getting a ZIP file.
    
    Args:
        api_url: Base URL of the FastAPI server
        image_path: Path to the image file to upload
        motion: Motion sequence name to use
    """
    
    print(f"Testing LAM API at {api_url}")
    
    # 1. Check if API is running
    try:
        response = requests.get(f"{api_url}/")
        print(f"API Status: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to API. Make sure the server is running.")
        return
    
    # 2. List available motions
    try:
        response = requests.get(f"{api_url}/motions")
        motions = response.json()["motions"]
        print(f"Available motions: {motions}")
        
        if motion not in motions:
            print(f"Warning: Motion '{motion}' not found. Using first available motion.")
            motion = motions[0] if motions else "nice"
    except Exception as e:
        print(f"Warning: Could not get motion list: {e}")
    
    # 3. Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    # 4. Upload image and process
    print(f"Uploading image: {image_path}")
    print(f"Using motion: {motion}")
    
    try:
        with open(image_path, 'rb') as image_file:
            files = {'image': (os.path.basename(image_path), image_file, 'image/png')}
            data = {'motion': motion}
            
            response = requests.post(
                f"{api_url}/process",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout
            )
        
        if response.status_code == 200:
            result = response.json()
            print("Processing successful!")
            print(f"Results: {result['results']}")
            
            # 5. Download files
            for file_type, file_path in result['results'].items():
                if file_path:
                    filename = os.path.basename(file_path)
                    download_url = f"{api_url}/download/{file_type}/{filename}"
                    
                    print(f"Downloading {file_type}: {filename}")
                    
                    download_response = requests.get(download_url)
                    if download_response.status_code == 200:
                        output_path = f"downloaded_{filename}"
                        with open(output_path, 'wb') as f:
                            f.write(download_response.content)
                        print(f"Saved to: {output_path}")
                    else:
                        print(f"Failed to download {filename}: {download_response.status_code}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("Error: Request timed out. Processing might take longer than expected.")
    except Exception as e:
        print(f"Error during processing: {e}")


def curl_example():
    """
    Print curl command examples for using the API.
    """
    print("\n" + "="*50)
    print("CURL EXAMPLES")
    print("="*50)
    
    print("\n1. Check API status:")
    print("curl http://localhost:8000/")
    
    print("\n2. List available motions:")
    print("curl http://localhost:8000/motions")
    
    print("\n3. Process an image:")
    print("curl -X POST http://localhost:8000/process \\")
    print("  -F 'image=@assets/sample_input/james.png' \\")
    print("  -F 'motion=nice'")
    
    print("\n4. Download a file:")
    print("curl -O http://localhost:8000/download/zip/uploaded_image_nice.zip")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LAM FastAPI")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--image", default="assets/sample_input/james.png", help="Image file to upload")
    parser.add_argument("--motion", default="nice", help="Motion sequence name")
    parser.add_argument("--curl-only", action="store_true", help="Only show curl examples")
    
    args = parser.parse_args()
    
    if args.curl_only:
        curl_example()
    else:
        test_api(args.api_url, args.image, args.motion)
        curl_example() 