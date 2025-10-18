"""
Inference script for RespiScope-AI
Handles prediction on new audio samples
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Union, Dict, List
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.audio_utils import AudioProcessor
from utils.config import get_config
from crnn_classifier import create_model


class RespiScopeInference:
    """Inference class for respiratory disease prediction"""
    
    def __init__(self,
                 model_path: str,
                 model_type: str = 'crnn',
                 config: dict = None,
                 device: str = 'cuda'):
        """
        Initialize inference
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model ('crnn' or 'transformer')
            config: Model configuration
            device: Device to use for inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # Load config
        if config is None:
            self.config = get_config()
        else:
            self.config = config
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=self.config.audio.sample_rate,
            duration=self.config.audio.duration,
            n_mels=self.config.audio.n_mels,
            n_fft=self.config.audio.n_fft,
            hop_length=self.config.audio.hop_length,
            fmin=self.config.audio.fmin,
            fmax=self.config.audio.fmax,
            top_db=self.config.audio.top_db
        )
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Class names
        self.class_names = self.config.model.class_names
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        print(f"Loading model from {model_path}")
        
        # Create model
        model = create_model(
            model_type=self.model_type,
            num_classes=self.config.model.num_classes,
            device=self.device
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        return model
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Preprocess audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor
        """
        # Load and preprocess audio
        audio = self.audio_processor.preprocess(
            audio_path,
            trim=True,
            normalize=True
        )
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        
        return audio_tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, 
                audio_path: str,
                return_probs: bool = True,
                top_k: int = None) -> Dict:
        """
        Make prediction on audio file
        
        Args:
            audio_path: Path to audio file
            return_probs: Whether to return probabilities
            top_k: Return top-k predictions
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess audio
        audio_tensor = self.preprocess_audio(audio_path)
        
        # Forward pass
        outputs = self.model(audio_tensor)
        
        # Get probabilities
        probs = torch.softmax(outputs, dim=1)
        probs = probs.cpu().numpy()[0]
        
        # Get prediction
        pred_idx = np.argmax(probs)
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        
        # Prepare result
        result = {
            'predicted_class': pred_class,
            'confidence': confidence,
            'class_index': int(pred_idx)
        }
        
        if return_probs:
            result['probabilities'] = {
                name: float(prob)
                for name, prob in zip(self.class_names, probs)
            }
        
        if top_k is not None:
            top_indices = np.argsort(probs)[::-1][:top_k]
            result['top_k_predictions'] = [
                {
                    'class': self.class_names[idx],
                    'confidence': float(probs[idx])
                }
                for idx in top_indices
            ]
        
        return result
    
    def predict_batch(self,
                     audio_paths: List[str],
                     batch_size: int = 8) -> List[Dict]:
        """
        Make predictions on multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            
            for audio_path in batch_paths:
                try:
                    result = self.predict(audio_path)
                    result['audio_path'] = audio_path
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {audio_path}: {str(e)}")
                    results.append({
                        'audio_path': audio_path,
                        'error': str(e)
                    })
        
        return results
    
    def predict_from_array(self, 
                          audio_array: np.ndarray,
                          sample_rate: int = 16000) -> Dict:
        """
        Make prediction from numpy array
        
        Args:
            audio_array: Audio as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with prediction results
        """
        # Resample if necessary
        if sample_rate != self.config.audio.sample_rate:
            import librosa
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sample_rate,
                target_sr=self.config.audio.sample_rate
            )
        
        # Preprocess
        audio_array = self.audio_processor.normalize_audio(audio_array)
        audio_array = self.audio_processor.pad_or_truncate(audio_array)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_array).float()
        audio_tensor = audio_tensor.unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(audio_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Get prediction
        pred_idx = np.argmax(probs)
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        
        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'probabilities': {
                name: float(prob)
                for name, prob in zip(self.class_names, probs)
            }
        }


def main():
    """Command-line interface for inference"""
    parser = argparse.ArgumentParser(description='RespiScope-AI Inference')
    parser.add_argument('--audio_path', type=str, required=True,
                       help='Path to audio file or directory')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='crnn',
                       choices=['crnn', 'transformer'],
                       help='Model type')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save predictions (JSON)')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Show top-k predictions')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory in batch mode')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = RespiScopeInference(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device
    )
    
    print("=" * 80)
    print("RespiScope-AI Inference")
    print("=" * 80)
    print(f"Model: {args.model_type}")
    print(f"Device: {inference.device}")
    print(f"Classes: {', '.join(inference.class_names)}")
    print("=" * 80)
    
    # Single file or batch processing
    if args.batch or os.path.isdir(args.audio_path):
        # Batch processing
        audio_dir = args.audio_path
        audio_files = []
        
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            audio_files.extend(Path(audio_dir).glob(f'*{ext}'))
        
        print(f"\nFound {len(audio_files)} audio files")
        print("Processing...")
        
        results = inference.predict_batch([str(f) for f in audio_files])
        
        # Print results
        print("\nPredictions:")
        print("-" * 80)
        for result in results:
            if 'error' not in result:
                filename = os.path.basename(result['audio_path'])
                print(f"{filename}: {result['predicted_class']} "
                      f"({result['confidence']:.2%})")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    else:
        # Single file processing
        print(f"\nProcessing: {args.audio_path}")
        
        result = inference.predict(
            args.audio_path,
            return_probs=True,
            top_k=args.top_k
        )
        
        # Print results
        print("\n" + "=" * 80)
        print("PREDICTION RESULTS")
        print("=" * 80)
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nProbabilities:")
        print("-" * 80)
        
        # Sort probabilities
        sorted_probs = sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for class_name, prob in sorted_probs:
            bar = '█' * int(prob * 50)
            print(f"{class_name:15s} {prob:6.2%} {bar}")
        
        if args.top_k and 'top_k_predictions' in result:
            print(f"\nTop-{args.top_k} Predictions:")
            print("-" * 80)
            for i, pred in enumerate(result['top_k_predictions'], 1):
                print(f"{i}. {pred['class']}: {pred['confidence']:.2%}")
        
        # Save result
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to {args.output}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
