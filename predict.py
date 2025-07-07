import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
import json
import csv
from pathlib import Path
from tqdm import tqdm
import timm
import pickle
import warnings
import argparse

warnings.filterwarnings('ignore')

class TestPredictor:
    def __init__(self, model_dir, device='cuda'):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.class_names = [
            "nose-right", "nose-left", "ear-right",
            "ear-left", "vc-open", "vc-closed", "throat"
        ]
        self.num_classes = len(self.class_names)

        print(f"Using device: {self.device}")

    def load_ensemble_models(self):
        print("Loading ensemble models...")

        with open(self.model_dir / 'ensemble_info.pkl', 'rb') as f:
            ensemble_info = pickle.load(f)

        models = []
        model_names = ensemble_info['models']
        weights = ensemble_info['weights']

        for i, model_name in enumerate(model_names):
            print(f"Loading model {i+1}/{len(model_names)}: {model_name}")

            base_name = "convnext_base.fb_in22k_ft_in1k"

            model = timm.create_model(base_name, pretrained=False, num_classes=self.num_classes)
            state_dict = torch.load(self.model_dir / f"ensemble_model_{i}.pt", map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            models.append({'model': model, 'weight': weights[i], 'name': model_name})

        print(f"Loaded {len(models)} models successfully.")
        return models

    def load_test_data(self, csv_path, img_dir):
        test_files = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    img_name = row[0].strip()
                    img_path = Path(img_dir) / img_name
                    if img_path.exists():
                        test_files.append(str(img_path))
                    else:
                        print(f"Warning: Image not found: {img_path}")

        print(f"Loaded {len(test_files)} test images.")
        return test_files

    def get_tta_transforms(self, img_size=224, n_aug=5):
        class ResizeOrPad:
            def __init__(self, min_size):
                self.min_size = min_size

            def __call__(self, img):
                w, h = img.size
                if w < self.min_size or h < self.min_size:
                    scale = self.min_size / min(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    return T.functional.resize(img, (new_h, new_w))
                return img

        transforms = [
            T.Compose([
                ResizeOrPad(img_size),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        ]

        import random
        for i in range(n_aug - 1):
            resize_delta = random.choice([-20, -10, 0, 10, 20])
            target_size = max(img_size, img_size + resize_delta)

            transforms.append(
                T.Compose([
                    ResizeOrPad(target_size + 20),
                    T.Resize((target_size + 10, target_size + 10)),
                    T.CenterCrop(img_size) if i % 2 == 0 else T.RandomCrop(img_size),
                    T.RandomApply([
                        T.ColorJitter(
                            brightness=random.uniform(0.1, 0.2),
                            contrast=random.uniform(0.1, 0.2),
                            saturation=random.uniform(0.05, 0.15),
                            hue=random.uniform(0.02, 0.05)
                        )
                    ], p=0.8),
                    T.RandomChoice([
                        T.GaussianBlur(3, sigma=(0.1, 0.5)),
                        nn.Identity()
                    ]),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            )
        return transforms

    def predict_single_image_tta(self, model, img_path, n_aug=5):
        image = Image.open(img_path).convert('RGB')
        tta_transforms = self.get_tta_transforms(n_aug=n_aug)

        aug_probs = []
        with torch.no_grad():
            for transform in tta_transforms:
                img_tensor = transform(image).unsqueeze(0).to(self.device)
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                aug_probs.append(probs)

        weights = torch.tensor([1.5] + [1.0] * (n_aug - 1)).to(self.device)
        weights = weights / weights.sum()

        final_probs = torch.zeros_like(aug_probs[0])
        for i, probs in enumerate(aug_probs):
            final_probs += probs * weights[i]

        return final_probs.cpu().numpy().squeeze()

    def ensemble_predict_batch(self, models, test_files, use_tta=True, batch_size=32):
        predictions = {}
        for img_path in tqdm(test_files, desc="Predicting"):
            img_name = Path(img_path).name
            all_model_probs = []
            model_weights = []

            for model_info in models:
                model = model_info['model']
                weight = model_info['weight']

                if use_tta:
                    probs = self.predict_single_image_tta(model, img_path, n_aug=3)
                else:
                    transform = T.Compose([
                        T.Resize((224, 224)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                    ])
                    image = Image.open(img_path).convert('RGB')
                    img_tensor = transform(image).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        output = model(img_tensor)
                        probs = F.softmax(output, dim=1).cpu().numpy().squeeze()

                all_model_probs.append(probs)
                model_weights.append(weight ** 2)

            model_weights = np.array(model_weights)
            model_weights /= model_weights.sum()

            final_probs = np.zeros_like(all_model_probs[0])
            for i, probs in enumerate(all_model_probs):
                final_probs += probs * model_weights[i]

            pred_class = np.argmax(final_probs)
            predictions[img_name] = int(pred_class)

        return predictions

    def predict_and_save(self, csv_path, img_dir, output_path, use_tta=True):
        models = self.load_ensemble_models()
        test_files = self.load_test_data(csv_path, img_dir)

        print("\nStarting prediction...")
        predictions = self.ensemble_predict_batch(models, test_files, use_tta=use_tta)

        print(f"\nSaving results to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)

        print(f"Saved {len(predictions)} predictions")

        print("\nSample predictions:")
        for i, (img_name, pred) in enumerate(list(predictions.items())[:5]):
            print(f"  {img_name}: {pred} ({self.class_names[pred]})")

        return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification Prediction using Ensemble Models")
    parser.add_argument("--model-dir", required=True, help="Path to the directory containing ensemble models")
    parser.add_argument("--csv-path", required=True, help="Path to the CSV file containing test image names")
    parser.add_argument("--img-dir", required=True, help="Path to the directory containing test images")
    parser.add_argument("--output-path", required=True, help="Path where predictions will be saved (JSON format)")
    parser.add_argument("--use-tta", action="store_true", default=True, help="Enable test-time augmentation (default: True)")
    parser.add_argument("--no-tta", dest="use_tta", action="store_false", help="Disable test-time augmentation")
    
    args = parser.parse_args()

    predictor = TestPredictor(args.model_dir)
    predictions = predictor.predict_and_save(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        output_path=args.output_path,
        use_tta=args.use_tta
    )

    print(f"Results saved to: {args.output_path}")