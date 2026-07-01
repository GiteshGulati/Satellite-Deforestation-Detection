from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

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

full_dataset = datasets.ImageFolder(
    root=r"C:\Users\Gitesh\PycharmProjects\PythonProject\data\EuroSAT\2750",
    transform=train_transform
)

# ----------------------------
# Split Dataset
# ----------------------------

train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_set, val_set, test_set = random_split(
    full_dataset,
    [train_size, val_size, test_size]
)

train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_set,
    batch_size=32,
    shuffle=False
)

model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

model = model.to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
num_epochs = 1

print("\n===================================")
print("Starting Model Training...")
print("===================================\n")

for epoch in range(num_epochs):

    print(f"\n========== Epoch {epoch + 1}/{num_epochs} ==========")

    # ----------------------------
    # Training Phase
    # ----------------------------
    print("Training...")

    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

        if batch_idx % 100 == 0:
            print(f"Processing Training Batch {batch_idx}/{len(train_loader)}")

        images = images.to("cuda")
        labels = labels.to("cuda")

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print("Training Complete.")

    # ----------------------------
    # Validation Phase
    # ----------------------------
    print("Starting Validation...")

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(val_loader):

            if batch_idx % 25 == 0:
                print(f"Validating Batch {batch_idx}/{len(val_loader)}")

            images = images.to("cuda")
            labels = labels.to("cuda")

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total

    print("\nEpoch Summary")
    print(f"Loss          : {running_loss / len(train_loader):.4f}")
    print(f"Val Accuracy  : {val_accuracy:.4f}")

print("\n===================================")
print("Training Finished Successfully!")
print("===================================")