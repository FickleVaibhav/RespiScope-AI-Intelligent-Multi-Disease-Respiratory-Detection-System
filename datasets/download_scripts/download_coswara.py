"""
Download Coswara Dataset
COVID-19 respiratory sound database
"""

import os
import argparse
import subprocess
import shutil
from tqdm import tqdm


def download_from_github():
    """Download Coswara dataset from GitHub"""
    print("Downloading Coswara dataset from GitHub...")
    
    repo_url = "https://github.com/iiscleap/Coswara-Data.git"
    output_dir = "data/raw/coswara"
    
    try:
        # Clone repository
        if os.path.exists(output_dir):
            print(f"Directory {output_dir} already exists.")
            response = input("Remove and re-download? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(output_dir)
            else:
                print("Using existing directory.")
                return True
        
        print(f"Cloning repository to {output_dir}...")
        subprocess.run(['git', 'clone', repo_url, output_dir], check=True)
        
        print("✅ Download complete!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during git clone: {e}")
        print("\nAlternative: Download manually from:")
        print("https://github.com/iiscleap/Coswara-Data")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def verify_dataset(data_dir: str):
    """Verify Coswara dataset"""
    print("\nVerifying dataset...")
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return False
    
    # Check for data directories
    expected_dirs = ['processed', 'Annotation']
    found_dirs = [d for d in expected_dirs if os.path.exists(os.path.join(data_dir, d))]
    
    print(f"Found directories: {found_dirs}")
    
    if len(found_dirs) > 0:
        print("✅ Dataset structure verified!")
        
        # Count audio files
        audio_count = 0
        for root, dirs, files in os.walk(data_dir):
            audio_count += len([f for f in files if f.endswith(('.wav', '.mp3', '.flac'))])
        
        print(f"📊 Total audio files: {audio_count}")
        return True
    else:
        print("⚠️ Expected directories not found")
        return False


def organize_coswara_data(input_dir: str, output_dir: str):
    """Organize Coswara data into a structured format"""
    print("\nOrganizing Coswara data...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Coswara has different audio types
    audio_types = [
        'cough-heavy',
        'cough-shallow',
        'breathing-deep',
        'breathing-shallow',
        'vowel-a',
        'vowel-e',
        'vowel-o',
        'counting-normal',
        'counting-fast'
    ]
    
    print("Audio types to process:", audio_types)
    print("\n⚠️ Note: Coswara dataset requires custom preprocessing")
    print("Please refer to the Coswara documentation for detailed structure")
    print("Link: https://github.com/iiscleap/Coswara-Data")


def main():
    parser = argparse.ArgumentParser(description='Download Coswara dataset')
    parser.add_argument('--output_dir', type=str, default='data/raw/coswara',
                       help='Output directory')
    parser.add_argument('--organize', action='store_true',
                       help='Organize data after download')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Coswara Dataset Download Script")
    print("=" * 80)
    print("\nNote: Coswara is primarily a COVID-19 dataset")
    print("Mapping to general respiratory conditions requires careful consideration")
    print("=" * 80)
    
    # Download
    success = download_from_github()
    
    if success:
        # Verify
        verify_dataset(args.output_dir)
        
        # Organize if requested
        if args.organize:
            organize_coswara_data(args.output_dir, args.output_dir + '_processed')
        
        print("\n" + "=" * 80)
        print("✅ Coswara dataset ready!")
        print(f"📁 Location: {args.output_dir}")
        print("\nNext steps:")
        print("1. Review Coswara documentation: https://github.com/iiscleap/Coswara-Data")
        print("2. Implement custom preprocessing for Coswara format")
        print("3. Map COVID status to respiratory conditions appropriately")
        print("=" * 80)


if __name__ == '__main__':
    main()
