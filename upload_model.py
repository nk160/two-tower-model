import wandb
import torch
from Version4 import TwoTowerModel
import os
import requests
from pathlib import Path
import sys
import hashlib

def get_file_hash(filename):
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_file_with_progress(url, filename, headers, expected_size=None):
    print(f"Downloading {filename}...")
    response = requests.get(url, headers=headers, stream=True, timeout=300)
    response.raise_for_status()
    
    # Create directory if it doesn't exist
    Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    
    # Get file size
    file_size = int(response.headers.get('content-length', 0))
    if expected_size and file_size != expected_size:
        print(f"Warning: Expected size {expected_size} but got {file_size}")
    
    # Download with progress
    with open(filename, 'wb') as f:
        if file_size == 0:
            print("Warning: File size unknown")
            f.write(response.content)
        else:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = min(100, int(100 * downloaded / file_size))
                    sys.stdout.write(f"\rProgress: {progress}% [{downloaded}/{file_size} bytes]")
                    sys.stdout.flush()
    
    # Verify file size
    actual_size = os.path.getsize(filename)
    if file_size != actual_size:
        raise Exception(f"Download incomplete! Expected {file_size} bytes but got {actual_size} bytes")
    
    print("\nDownload complete! Verifying file...")
    return get_file_hash(filename)

def upload_to_wandb():
    print("Logging into W&B...")
    wandb.login()
    
    print("Initializing W&B run...")
    api = wandb.Api()
    run = api.run("nigelkiernan-lpt-advisory/two-tower-search/xgwqz248")
    
    # List available files
    print("Available files in run:")
    files = run.files()
    for file in files:
        print(f"- {file.name} ({file.size} bytes)")
    
    # Download the checkpoint file directly
    print("\nDownloading checkpoint file...")
    checkpoint_path = 'checkpoints/checkpoint_epoch_1.pt'
    for file in files:
        if file.name == checkpoint_path:
            headers = {'Authorization': f'Bearer {api.api_key}'}
            # Remove old file if it exists
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            file_hash = download_file_with_progress(file.url, checkpoint_path, headers, file.size)
            print(f"File hash: {file_hash}")
            break
    
    print("Loading checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("File may be corrupted. Please try downloading again.")
        return
    
    # Initialize model with the saved vocabulary
    model = TwoTowerModel(
        vocab_size=len(checkpoint['vocab']),
        vocab=checkpoint['vocab']
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Saving new artifact...")
    # Save as a new artifact
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': checkpoint['vocab'],
        'config': {
            'embedding_dim': 300,
            'hidden_dim': 768,
            'output_dim': 384
        }
    }, "model.pth")
    
    print("Creating new artifact...")
    with wandb.init(project="two-tower-search", job_type="upload") as upload_run:
        artifact = wandb.Artifact('model-checkpoint', type='model', version='v1')
        artifact.add_file("model.pth")
        
        print("Logging artifact...")
        upload_run.log_artifact(artifact)
    
    print("Upload complete!")

if __name__ == "__main__":
    upload_to_wandb() 