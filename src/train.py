import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "EuroSAT", "2750")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "satellite_classifier.pth")

NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Transforms
# ----------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class TransformSubset(Subset):
    """Wrapper around Subset that applies a custom transform,
    overriding the parent dataset's transform. This fixes the issue
    where random_split shares the parent's transform across all subsets."""

    def __init__(self, subset, transform):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx):
        # Get the original image and label without the parent's transform
        original_transform = self.dataset.transform
        self.dataset.transform = self.transform
        item = self.dataset[self.indices[idx]]
        self.dataset.transform = original_transform
        return item

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]



# ----------------------------
# Load Dataset
# ----------------------------

print(f"Loading EuroSAT dataset from: {DATA_DIR}")
print(f"Using device: {DEVICE}")

full_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=train_transform  # Default; overridden per-subset below
)

class_names = full_dataset.classes
print(f"Found {len(full_dataset)} images across {len(class_names)} classes:")
for i, name in enumerate(class_names):
    print(f"  [{i}] {name}")

# ----------------------------
# Split Dataset (80/10/10)
# ----------------------------

train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_subset, val_subset, test_subset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # Reproducible splits
)

# Apply correct transforms: train augmentation for train, no augmentation for val/test
train_set = TransformSubset(train_subset, train_transform)
val_set = TransformSubset(val_subset, test_transform)
test_set = TransformSubset(test_subset, test_transform)

print(f"\nDataset split: {train_size} train / {val_size} val / {test_size} test")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# Build Model (ResNet50 Transfer Learning)
# ----------------------------

print("\nLoading pre-trained ResNet50...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freeze all backbone layers — only train the final classifier
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)

model = model.to(DEVICE)
print(f"Model ready. Training {sum(p.numel() for p in model.fc.parameters()):,} parameters (classifier head only).")

# ----------------------------
# Loss, Optimizer
# ----------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# ----------------------------
# Training Loop
# ----------------------------

print("\n" + "=" * 50)
print("  STARTING MODEL TRAINING")
print("=" * 50)

best_val_accuracy = 0.0

for epoch in range(NUM_EPOCHS):

    print(f"\n{'-' * 50}")
    print(f"  Epoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"{'-' * 50}")

    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    train_bar = tqdm(train_loader, desc="  Training", unit="batch", leave=True)
    for images, labels in train_bar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    train_accuracy = train_correct / train_total
    avg_loss = running_loss / len(train_loader)

    # --- Validation Phase ---
    model.eval()
    correct = 0
    total = 0

    val_bar = tqdm(val_loader, desc="  Validating", unit="batch", leave=True)
    with torch.no_grad():
        for images, labels in val_bar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total

    print(f"\n  Summary:")
    print(f"    Train Loss     : {avg_loss:.4f}")
    print(f"    Train Accuracy : {train_accuracy:.4f}")
    print(f"    Val Accuracy   : {val_accuracy:.4f}")

    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'class_names': class_names,
        }, MODEL_PATH)
        print(f"    [SAVED] New best model saved! (val_accuracy={val_accuracy:.4f})")

# ----------------------------
# Final Summary
# ----------------------------

print("\n" + "=" * 50)
print("  TRAINING COMPLETE")
print("=" * 50)
print(f"  Best Validation Accuracy: {best_val_accuracy:.4f}")
print(f"  Model saved to: {MODEL_PATH}")
print("=" * 50)