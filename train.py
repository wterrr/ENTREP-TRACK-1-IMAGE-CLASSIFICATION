#!/usr/bin/env python3
"""
Main training script for endoscopy image classification
"""

import argparse
import sys
from pathlib import Path

from src.classifier import EndoscopyClassifier


def main():
    parser = argparse.ArgumentParser(description='Train endoscopy image classifier')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--project_name', type=str, default='endoscopy_95',
                        help='Project name for output directory')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path '{args.data_path}' does not exist")
        sys.exit(1)
    
    print(f"Starting training with:")
    print(f"  Data path: {args.data_path}")
    print(f"  Project name: {args.project_name}")
    print(f"  Test size: {args.test_size}")
    print("-" * 60)
    
    # Create classifier and run training pipeline
    classifier = EndoscopyClassifier(
        data_path=args.data_path,
        project_name=args.project_name,
        test_size=args.test_size
    )
    
    results = classifier.run_pipeline()
    
    if results:
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Final accuracy: {results['final_accuracy']:.4f} ({results['final_accuracy']*100:.2f}%)")
        print(f"Ensemble accuracy: {results['ensemble_accuracy']:.4f} ({results['ensemble_accuracy']*100:.2f}%)")
        print(f"Best single model accuracy: {results['best_single_accuracy']:.4f} ({results['best_single_accuracy']*100:.2f}%)")
        print(f"Number of models in ensemble: {results['num_models']}")
        print(f"Results saved to: results_{args.project_name}/")
    else:
        print("\nTraining failed. Check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()