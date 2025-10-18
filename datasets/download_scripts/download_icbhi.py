"""
Download ICBHI 2017 Respiratory Sound Database
Dataset: https://bhichallenge.med.auth.gr/
"""

import os
import requests
import zipfile
from tqdm import tqdm
import kaggle
import argparse


def download_from_kaggle():
    """Download ICBHI dataset from Kaggle"""
    print("Downloading ICBHI 2017 dataset from Kaggle...")
    print("Note: You need Kaggle API credentials (~/.kaggle/kaggle.json)")
    
    try:
        # Download using Kaggle API
        kaggle.api.dataset_download_files(
            'vbookshelf/respiratory-sound-database',
            path='data/raw/icbhi',
            unzip=True
        )
        print("✅ Download complete!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Create API token (downloads kaggle.json)")
        print("3. Place in ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False


def download_from_url(url: str, output_dir: str):
    """Download dataset from direct URL"""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, 'icbhi_dataset.zip')
    
    print(f"Downloading from {url}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    
    print("Extracting files...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    os.remove(filename)
    print("✅ Download and extraction complete!")


def verify_dataset(data_dir: str):
    """Verify dataset integrity"""
    print("\nVerifying dataset...")
    
    audio_dir = os.path.join(data_dir, 'audio_and_txt_files')
    
    if not os.path.exists(audio_dir):
        print("❌ Dataset directory not found!")
        return False
    
    # Count audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    txt_files = [f for f in os.listdir(audio_dir) if f.endswith('.txt')]
    
    print(f"✅ Found {len(audio_files)} audio files (.wav)")
    print(f"✅ Found {len(txt_files)} annotation files (.txt)")
    
    if len(audio_files) > 0:
        print("\n📊 Dataset Statistics:")
        print(f"  - Total recordings: {len(audio_files)}")
        print(f"  - Expected: ~920 recordings")
        print(f"  - Status: {'✅ Complete' if len(audio_files) >= 900 else '⚠️ Incomplete'}")
        return True
    else:
        print("❌ No audio files found!")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download ICBHI 2017 dataset')
    parser.add_argument('--output_dir', type=str, default='data/raw/icbhi',
                       help='Output directory')
    parser.add_argument('--method', type=str, default='kaggle',
                       choices=['kaggle', 'url'],
                       help='Download method')
    parser.add_argument('--url', type=str, default=None,
                       help='Direct download URL (if using url method)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ICBHI 2017 Respiratory Sound Database - Download Script")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.method == 'kaggle':
        success = download_from_kaggle()
    elif args.method == 'url' and args.url:
        download_from_url(args.url, args.output_dir)
        success = True
    else:
        print("❌ Please specify download method and URL (if needed)")
        return
    
    if success:
        verify_dataset(args.output_dir)
        
        print("\n" + "=" * 80)
        print("✅ ICBHI dataset ready!")
        print(f"📁 Location: {args.output_dir}")
        print("\nNext steps:")
        print("1. Run preprocessing: python datasets/preprocessing/audio_preprocessing.py")
        print("2. Prepare splits: python datasets/preprocessing/prepare_splits.py")
        print("=" * 80)


if __name__ == '__main__':
    main()
