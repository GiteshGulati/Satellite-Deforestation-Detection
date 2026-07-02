"""
evaluate.py — Evaluate the trained satellite image classifier.

Loads the saved model, runs inference on the test set, and generates:
  1. Classification report (precision, recall, F1-score per class)
  2. Confusion matrix heatmap saved to outputs/confusion_matrix.png
  3. Per-class analysis with commentary on model strengths and weaknesses
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "EuroSAT", "2750")
MODEL_PATH = os.path.join(BASE_DIR, "models", "satellite_classifier.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

NUM_CLASSES = 10
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names matching EuroSAT folder order (alphabetical)
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbVeg', 'Highway', 'Industrial',
    'Pasture', 'PermCrop', 'Residential', 'River', 'SeaLake'
]

# ----------------------------
# Transforms (no augmentation for evaluation)
# ----------------------------

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class TransformSubset(Subset):
    """Wrapper that applies a custom transform to a Subset."""

    def __init__(self, subset, transform):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx):
        original_transform = self.dataset.transform
        self.dataset.transform = self.transform
        item = self.dataset[self.indices[idx]]
        self.dataset.transform = original_transform
        return item

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]



def load_model():
    """Load the trained model from checkpoint."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No saved model found at {MODEL_PATH}")
        print("Please run train.py first to train and save the model.")
        sys.exit(1)

    print(f"Loading model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    print(f"  Trained for {checkpoint['epoch']} epoch(s)")
    print(f"  Best val accuracy: {checkpoint['val_accuracy']:.4f}")
    print(f"  Classes: {checkpoint.get('class_names', CLASS_NAMES)}")

    return model


def get_test_loader():
    """Recreate the test set using the same split as training."""
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=test_transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    _, _, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Same seed as training
    )

    test_set = TransformSubset(test_subset, test_transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Test set: {test_size} images")
    return test_loader


def run_inference(model, test_loader):
    """Run the model on the entire test set, collecting predictions and labels."""
    all_preds = []
    all_labels = []

    print("\nRunning inference on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Testing", unit="batch"):
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def print_classification_report(all_labels, all_preds):
    """Print the sklearn classification report."""
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    print(report)
    return report


def generate_confusion_matrix(all_labels, all_preds):
    """Generate and save a confusion matrix heatmap."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        linecolor='gray',
        square=True
    )
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('EuroSAT Classification — Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nConfusion matrix saved to: {output_path}")
    return cm


def print_per_class_analysis(cm, all_labels, all_preds):
    """Print analysis of model performance per class."""
    print("\n" + "=" * 60)
    print("  PER-CLASS ANALYSIS")
    print("=" * 60)

    accuracies = cm.diagonal() / cm.sum(axis=1)
    sorted_indices = np.argsort(accuracies)[::-1]

    print("\n  Ranked by Accuracy:")
    print(f"  {'Class':<22} {'Accuracy':>10} {'Support':>10}")
    print(f"  {'-' * 44}")
    for idx in sorted_indices:
        bar_count = int(accuracies[idx] * 20)
        bar = '#' * bar_count + '-' * (20 - bar_count)
        print(f"  {CLASS_NAMES[idx]:<22} {accuracies[idx]:>9.1%}  [{bar}]")

    # --- Highlight strong classes ---
    print("\n  [+] STRONG CLASSES (typically >95% accuracy):")
    strong = [CLASS_NAMES[i] for i in sorted_indices if accuracies[i] >= 0.95]
    if strong:
        for name in strong:
            print(f"    * {name} -- distinct visual patterns make classification easy")
    else:
        print("    * None achieved >95% accuracy in this run")

    # --- Highlight confused classes ---
    print("\n  [-] COMMONLY CONFUSED PAIRS:")
    confusion_pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i][j] > 0:
                confusion_rate = cm[i][j] / cm[i].sum()
                if confusion_rate >= 0.05:  # 5%+ confusion rate
                    confusion_pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], cm[i][j], confusion_rate))

    confusion_pairs.sort(key=lambda x: x[3], reverse=True)
    if confusion_pairs:
        for true_cls, pred_cls, count, rate in confusion_pairs[:8]:
            print(f"    * {true_cls} -> {pred_cls}: {count} misclassified ({rate:.1%})")
    else:
        print("    * No significant confusion pairs detected")

    # --- Known patterns from the guide ---
    print("\n  [INFO] Expected Patterns (from EuroSAT literature):")
    print("    * Forest & SeaLake: near-perfect -- very distinct spectral signatures")
    print("    * Highway & River: often confused -- both appear as thin linear features")
    print("    * Pasture, HerbVeg, AnnualCrop: overlap -- all green ground cover from above")


def save_report(report, cm, all_labels, all_preds):
    """Save evaluation results to a text file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")

    overall_accuracy = np.sum(all_preds == all_labels) / len(all_labels)

    with open(report_path, 'w') as f:
        f.write("EuroSAT Satellite Image Classification — Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Test Accuracy: {overall_accuracy:.4f} ({overall_accuracy:.1%})\n")
        f.write(f"Total Test Samples: {len(all_labels)}\n")
        f.write(f"Correct Predictions: {np.sum(all_preds == all_labels)}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix (rows=true, cols=predicted):\n")
        f.write(str(cm) + "\n")

    print(f"Full report saved to: {report_path}")


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  EuroSAT MODEL EVALUATION")
    print("=" * 60)
    print(f"  Device: {DEVICE}\n")

    # 1. Load model
    model = load_model()

    # 2. Load test data
    test_loader = get_test_loader()

    # 3. Run inference
    all_preds, all_labels = run_inference(model, test_loader)

    # 4. Classification report
    report = print_classification_report(all_labels, all_preds)

    # 5. Confusion matrix
    cm = generate_confusion_matrix(all_labels, all_preds)

    # 6. Per-class analysis
    print_per_class_analysis(cm, all_labels, all_preds)

    # 7. Save report
    save_report(report, cm, all_labels, all_preds)

    # 8. Overall accuracy
    overall_acc = np.sum(all_preds == all_labels) / len(all_labels)
    print("\n" + "=" * 60)
    print(f"  OVERALL TEST ACCURACY: {overall_acc:.4f} ({overall_acc:.1%})")
    print("=" * 60)
