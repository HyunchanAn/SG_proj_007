import os
import urllib.request
import subprocess
import sys

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"File {dest} already exists. Skipping.")
        return
    print(f"Downloading {url} to {dest}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download: {e}")
        sys.exit(1)

def main():
    os.makedirs("models/sam2", exist_ok=True)
    os.makedirs("models/depth_anything_v2", exist_ok=True)
    
    sam2_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    sam2_dest = "models/sam2/sam2_hiera_large.pt"
    download_file(sam2_url, sam2_dest)
    
    depth_url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
    depth_dest = "models/depth_anything_v2/depth_anything_v2_vitl.pth"
    download_file(depth_url, depth_dest)
    
    print("Starting Streamlit...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()
