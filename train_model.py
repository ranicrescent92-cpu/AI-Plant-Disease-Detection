"""
Train a ResNet18 image classifier on a PlantVillage-style folder dataset.

Expected dataset layout:
data/PlantVillage/
    Apple___Apple_scab/
        image1.jpg
        ...
    Apple___Black_rot/
        ...
    Tomato___healthy/
        ...

Download the RGB/color PlantVillage dataset and place the class folders inside
data/PlantVillage. The script creates a random train/validation split.
"""
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "PlantVillage"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}\n"
            "Download PlantVillage and place its class folders inside data/PlantVillage."
        )

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ImageFolder sorts class names alphabetically. Save this mapping for app.py.
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    if len(full_dataset.classes) < 2:
        raise ValueError("At least two class folders are required.")
    class_names = full_dataset.classes
    with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    generator = torch.Generator().manual_seed(SEED)
    val_size = max(1, int(len(full_dataset) * 0.2))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Use deterministic non-augmented transforms for validation.
    val_set.dataset = datasets.ImageFolder(DATA_DIR, transform=eval_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Images: {len(full_dataset)} | Classes: {len(class_names)}")
    print("Classes:", class_names)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Transfer learning: use ImageNet pretrained ResNet18, replace final layer.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    for param in model.fc.parameters():
        param.requires_grad = True
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    best_accuracy = 0.0
    save_path = MODEL_DIR / "plant_disease_resnet18.pth"

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = 100 * correct / max(total, 1)
        val_acc = 100 * val_correct / max(val_total, 1)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss={running_loss / max(total, 1):.4f} | "
            f"train_acc={train_acc:.2f}% | val_acc={val_acc:.2f}%"
        )
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print(f"Training complete. Best validation accuracy: {best_accuracy:.2f}%")
    print("Now run: streamlit run app.py")

if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)
    main()
