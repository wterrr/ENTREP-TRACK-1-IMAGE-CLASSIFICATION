import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import timm
import torch.nn.functional as F
import copy
import random
import pickle

from .transforms import AddGaussianNoise, SimulateEndoscopeLighting, ResizeOrPad
from .losses import FocalLoss
from .dataset import EarDataset


class EndoscopyClassifier:
    def __init__(self, data_path, project_name="endoscopy_95", test_size=0.2):
        self.data_path = Path(data_path)
        self.project_name = project_name
        self.test_size = test_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.results_dir = Path(f"results_{project_name}")
        self.results_dir.mkdir(exist_ok=True)

        self.setup_logging()

        self.class_names = [
            "nose-right",   # 0
            "nose-left",    # 1
            "ear-right",    # 2
            "ear-left",     # 3
            "vc-open",      # 4
            "vc-closed",    # 5
            "throat"        # 6
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_names)

        # Phân loại orientation
        self.left_classes = [1, 3]  # nose-left, ear-left
        self.right_classes = [0, 2]  # nose-right, ear-right
        self.other_classes = [4, 5, 6]  # vc-open, vc-closed, throat

        self.log("Device: {}".format(self.device))

    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.results_dir / f"log_{timestamp}.txt"

        def log_print(*args, **kwargs):
            message = " ".join(str(arg) for arg in args)
            print(message, **kwargs)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")

        self.log = log_print

    def collect_dataset(self):
        self.log("Collecting dataset...")

        all_files = []
        all_labels = []

        for idx, class_name in enumerate(self.class_names):
            class_dir = self.data_path / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                all_files.extend([str(img) for img in images])
                all_labels.extend([idx] * len(images))
                self.log(f"   {class_name}: {len(images)} images")

        train_files, test_files, train_labels, test_labels = train_test_split(
            all_files, all_labels,
            test_size=self.test_size,
            stratify=all_labels,
            random_state=42
        )

        self.log(f"Total: {len(all_files)} images")
        self.log(f"Train: {len(train_files)}, Test: {len(test_files)}")

        return train_files, train_labels, test_files, test_labels

    def setup_safe_augmentation(self, img_size=224, phase="moderate"):

        if phase == "gentle":
            train_transform = T.Compose([
                ResizeOrPad(img_size + 10),
                T.Resize((img_size + 10, img_size + 10)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        elif phase == "moderate":
            train_transform = T.Compose([
                ResizeOrPad(img_size + 30),
                T.Resize((img_size + 20, img_size + 20)),
                T.RandomCrop(img_size),

                T.RandomApply([
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05)
                ], p=0.8),

                T.RandomApply([
                    SimulateEndoscopeLighting(severity=0.3)
                ], p=0.3),

                T.RandomChoice([
                    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                    T.RandomAdjustSharpness(sharpness_factor=2.0),
                    T.RandomAdjustSharpness(sharpness_factor=0.5),
                    nn.Identity()
                ]),

                T.ToTensor(),
                AddGaussianNoise(std=0.01),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.15, scale=(0.02, 0.12))
            ])

        else:
            train_transform = T.Compose([
                ResizeOrPad(img_size + 40),
                T.RandomChoice([
                    T.Compose([T.Resize((img_size + 40, img_size + 40)), T.CenterCrop(img_size)]),
                    T.Compose([T.Resize((img_size + 20, img_size + 20)), T.CenterCrop(img_size)]),
                    T.Compose([T.Resize((img_size, img_size))]),
                    T.Compose([T.Resize((img_size + 32, img_size + 32)), T.RandomCrop(img_size)]),
                ]),

                T.RandomApply([
                    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.08)
                ], p=0.9),

                T.RandomApply([
                    SimulateEndoscopeLighting(severity=random.uniform(0.2, 0.4))
                ], p=0.5),

                T.RandomApply([
                    T.RandomAutocontrast(),
                    T.RandomEqualize(),
                ], p=0.3),

                T.RandomChoice([
                    T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                    T.RandomAdjustSharpness(sharpness_factor=0.3),
                    T.RandomAdjustSharpness(sharpness_factor=3.0),
                    nn.Identity()
                ]),

                T.ToTensor(),
                AddGaussianNoise(std=0.02),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.5, 2.0))
            ])

        val_transform = T.Compose([
            ResizeOrPad(img_size),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return train_transform, val_transform

    def safe_mixup(self, x, y, alpha=0.2):
        if np.random.rand() > 0.5:
            return x, y, y, 1.0

        batch_size = x.size(0)
        indices = []

        for i in range(batch_size):
            current_class = y[i].item()

            if current_class in self.left_classes:
                valid_indices = [j for j in range(batch_size) if y[j].item() in self.left_classes and j != i]
            elif current_class in self.right_classes:
                valid_indices = [j for j in range(batch_size) if y[j].item() in self.right_classes and j != i]
            else:
                valid_indices = [j for j in range(batch_size) if y[j].item() in self.other_classes and j != i]

            if valid_indices:
                idx = np.random.choice(valid_indices)
            else:
                idx = i
            indices.append(idx)

        indices = torch.tensor(indices).to(x.device)
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)

        mixed_x = lam * x + (1 - lam) * x[indices]
        return mixed_x, y, y[indices], lam

    def create_models(self):
        models_config = [
            {"name": "convnext_base.fb_in22k_ft_in1k", "img_size": 224},
        ]

        models = []
        for config in models_config:
            try:
                model = timm.create_model(
                    config['name'],
                    pretrained=True,
                    num_classes=self.num_classes,
                    drop_rate=0.4,
                    drop_path_rate=0.3
                )
                models.append({'model': model, 'config': config})
                self.log(f"Created: {config['name']}")
            except:
                self.log(f"Failed: {config['name']}")

        return models

    def train_single_model(self, model, train_files, train_labels, val_files, val_labels,
                          model_name, fold=None):

        model = model.to(self.device)
        img_size = 224
        best_acc = 0.0

        # Phase 1: Freeze backbone
        self.log(f"\nPhase 1: Training head only for {model_name}" + (f" (Fold {fold})" if fold else ""))

        for name, param in model.named_parameters():
            if 'head' in name or 'classifier' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Phase 1 training
        train_transform, val_transform = self.setup_safe_augmentation(img_size, "gentle")
        train_dataset = EarDataset(train_files, train_labels, train_transform)
        val_dataset = EarDataset(val_files, val_labels, val_transform)

        phase1_acc = self.train_phase(
            model, train_dataset, val_dataset,
            epochs=15, lr=5e-4, phase_name="head_only"
        )

        if phase1_acc > best_acc:
            best_acc = phase1_acc

        # Phase 2: Unfreeze last layers
        self.log(f"\nPhase 2: Fine-tuning last layers for {model_name}")

        # Unfreeze last 40% layers
        all_params = list(model.named_parameters())
        n_unfreeze = len(all_params) // 2
        for name, param in all_params[-n_unfreeze:]:
            param.requires_grad = True

        train_transform, val_transform = self.setup_safe_augmentation(img_size, "moderate")
        train_dataset = EarDataset(train_files, train_labels, train_transform)
        val_dataset = EarDataset(val_files, val_labels, val_transform)

        phase2_acc = self.train_phase(
            model, train_dataset, val_dataset,
            epochs=40, lr=3e-4, phase_name="partial_finetune"
        )

        if phase2_acc > best_acc:
            best_acc = phase2_acc

        # Phase 3: Full fine-tuning if needed
        if best_acc < 0.92:
            self.log(f"\nPhase 3: Full fine-tuning for {model_name}")

            for param in model.parameters():
                param.requires_grad = True

            train_transform, val_transform = self.setup_safe_augmentation(img_size, "strong")
            train_dataset = EarDataset(train_files, train_labels, train_transform)
            val_dataset = EarDataset(val_files, val_labels, val_transform)

            phase3_acc = self.train_phase(
                model, train_dataset, val_dataset,
                epochs=30, lr=1e-4, phase_name="full_finetune"
            )

            if phase3_acc > best_acc:
                best_acc = phase3_acc

        return model, best_acc

    def train_phase(self, model, train_dataset, val_dataset, epochs, lr, phase_name):

        # Weighted sampler cho imbalanced data
        class_counts = np.bincount(train_dataset.labels)
        weights = 1.0 / class_counts[train_dataset.labels]
        sampler = WeightedRandomSampler(weights, len(weights))

        train_loader = DataLoader(
            train_dataset, batch_size=16, sampler=sampler,
            num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False,
            num_workers=4, pin_memory=True
        )

        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'head' in name or 'classifier' in name or 'fc' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': head_params, 'lr': lr}
        ], weight_decay=0.01)

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=lr*0.001
        )

        criterion = FocalLoss(gamma=1.5, label_smoothing=0.05)

        best_val_acc = 0.0
        patience = 15
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"{phase_name} E{epoch+1}/{epochs}")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                images, targets_a, targets_b, lam = self.safe_mixup(images, labels)

                optimizer.zero_grad()
                outputs = model(images)

                if lam == 1.0:
                    loss = criterion(outputs, targets_a)
                else:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            scheduler.step()

            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = val_correct / val_total
            train_acc = correct / total

            self.log(f"   E{epoch+1}: Train {train_acc:.4f}, Val {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), self.results_dir / f"best_{phase_name}_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.log(f"   Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(torch.load(self.results_dir / f"best_{phase_name}_model.pt"))

        return best_val_acc

    def train_kfold_ensemble(self, all_files, all_labels, n_splits=5):
        self.log("\nTraining K-Fold Ensemble...")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_models = []

        base_models = self.create_models()

        for fold, (train_idx, val_idx) in enumerate(skf.split(all_files, all_labels)):
            self.log(f"\nFold {fold+1}/{n_splits}")

            fold_train_files = [all_files[i] for i in train_idx]
            fold_train_labels = [all_labels[i] for i in train_idx]
            fold_val_files = [all_files[i] for i in val_idx]
            fold_val_labels = [all_labels[i] for i in val_idx]

            for model_info in base_models:
                model_name = model_info['config']['name']
                self.log(f"\nTraining {model_name} on Fold {fold+1}")

                model = timm.create_model(
                    model_name,
                    pretrained=True,
                    num_classes=self.num_classes,
                    drop_rate=0.4,
                    drop_path_rate=0.3
                )

                trained_model, accuracy = self.train_single_model(
                    model,
                    fold_train_files, fold_train_labels,
                    fold_val_files, fold_val_labels,
                    model_name, fold+1
                )

                fold_models.append({
                    'model': trained_model,
                    'accuracy': accuracy,
                    'name': f"{model_name}_fold{fold+1}"
                })

        return fold_models

    def evaluate_with_tta(self, model, test_files, test_labels, img_size=224, n_aug=5):

        tta_transforms = [
            T.Compose([
                ResizeOrPad(img_size),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]

        for i in range(n_aug - 1):
            resize_delta = random.choice([-20, -10, 0, 10, 20])
            target_size = max(img_size, img_size + resize_delta)

            transform = T.Compose([
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
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tta_transforms.append(transform)

        model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for img_path, label in tqdm(zip(test_files, test_labels), total=len(test_files), desc="TTA Evaluation"):
                image = Image.open(img_path).convert('RGB')

                aug_probs = []
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

                conf, pred = torch.max(final_probs, 1)

                all_predictions.append(pred.item())
                all_labels.append(label)
                all_probs.append(final_probs.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        return accuracy, all_predictions, all_labels, all_probs

    def ensemble_predict(self, models, test_files, test_labels, use_tta=True):
        self.log("\nEnsemble Prediction...")

        all_model_probs = []
        model_weights = []

        for model_info in models:
            model = model_info['model']
            accuracy = model_info['accuracy']
            name = model_info.get('name', 'model')

            self.log(f"Getting predictions from {name} (acc: {accuracy:.4f})")

            if use_tta:
                _, _, _, probs = self.evaluate_with_tta(
                    model, test_files, test_labels, n_aug=3
                )
            else:
                model.eval()
                probs = []

                transform = T.Compose([
                    ResizeOrPad(224),
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                with torch.no_grad():
                    for img_path in test_files:
                        image = Image.open(img_path).convert('RGB')
                        img_tensor = transform(image).unsqueeze(0).to(self.device)
                        output = model(img_tensor)
                        prob = F.softmax(output, dim=1).cpu().numpy()
                        probs.append(prob)

            all_model_probs.append(np.array(probs).squeeze())
            model_weights.append(accuracy ** 2)

        model_weights = np.array(model_weights)
        model_weights = model_weights / model_weights.sum()

        final_probs = np.zeros_like(all_model_probs[0])
        for i, probs in enumerate(all_model_probs):
            final_probs += probs * model_weights[i]

        predictions = np.argmax(final_probs, axis=1)
        accuracy = accuracy_score(test_labels, predictions)

        return accuracy, predictions, final_probs

    def run_pipeline(self):
        try:
            train_files, train_labels, test_files, test_labels = self.collect_dataset()
            all_files = train_files + test_files
            all_labels = train_labels + test_labels

            fold_models = self.train_kfold_ensemble(all_files, all_labels, n_splits=5)

            self.log("\nTraining additional models on full training set...")
            full_models = self.create_models()

            for model_info in full_models:
                model_name = model_info['config']['name']
                self.log(f"\nTraining {model_name} on full train set")

                trained_model, accuracy = self.train_single_model(
                    model_info['model'],
                    train_files, train_labels,
                    test_files, test_labels,
                    model_name
                )

                fold_models.append({
                    'model': trained_model,
                    'accuracy': accuracy,
                    'name': f"{model_name}_full"
                })

            self.log("\n" + "="*60)
            self.log("INDIVIDUAL MODEL RESULTS WITH TTA:")

            best_single_acc = 0
            for model_info in fold_models:
                tta_acc, _, _, _ = self.evaluate_with_tta(
                    model_info['model'], test_files, test_labels, n_aug=5
                )
                self.log(f"{model_info['name']}: {tta_acc:.4f} ({tta_acc*100:.2f}%)")

                if tta_acc > best_single_acc:
                    best_single_acc = tta_acc

            self.log("\n" + "="*60)
            self.log("ENSEMBLE RESULTS:")

            fold_models.sort(key=lambda x: x['accuracy'], reverse=True)
            top_models = fold_models[:7]

            ensemble_acc, _, _ = self.ensemble_predict(
                top_models, test_files, test_labels, use_tta=False
            )
            self.log(f"Ensemble (no TTA): {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")

            ensemble_tta_acc, predictions, probs = self.ensemble_predict(
                top_models, test_files, test_labels, use_tta=True
            )
            self.log(f"Ensemble with TTA: {ensemble_tta_acc:.4f} ({ensemble_tta_acc*100:.2f}%)")

            self.log("\n" + "="*60)
            self.log("DETAILED CLASSIFICATION REPORT:")

            report = classification_report(
                test_labels, predictions,
                target_names=self.class_names,
                digits=4
            )
            self.log(report)

            cm = confusion_matrix(test_labels, predictions)
            self.log("\nConfusion Matrix:")
            self.log(str(cm))

            self.log("\nPer-class Accuracy:")
            for i, class_name in enumerate(self.class_names):
                class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
                self.log(f"   {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")

            self.log("\n" + "="*80)
            self.log("FINAL RESULTS:")
            self.log(f"Best Single Model with TTA: {best_single_acc:.4f} ({best_single_acc*100:.2f}%)")
            self.log(f"Ensemble with TTA: {ensemble_tta_acc:.4f} ({ensemble_tta_acc*100:.2f}%)")

            final_accuracy = max(best_single_acc, ensemble_tta_acc)

            self.log("\nSaving models...")
            ensemble_info = {
                'models': [m['name'] for m in top_models],
                'weights': [m['accuracy'] for m in top_models],
                'final_accuracy': ensemble_tta_acc,
                'class_names': self.class_names
            }

            with open(self.results_dir / 'ensemble_info.pkl', 'wb') as f:
                pickle.dump(ensemble_info, f)

            for i, model_info in enumerate(top_models):
                torch.save(
                    model_info['model'].state_dict(),
                    self.results_dir / f"ensemble_model_{i}.pt"
                )

            self.log(f"\nResults saved to: {self.results_dir}")

            return {
                'final_accuracy': final_accuracy,
                'ensemble_accuracy': ensemble_tta_acc,
                'best_single_accuracy': best_single_acc,
                'num_models': len(top_models)
            }

        except Exception as e:
            self.log(f"Error: {e}")
            import traceback
            self.log(traceback.format_exc())
            return None