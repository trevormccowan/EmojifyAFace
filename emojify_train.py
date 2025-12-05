import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt


###############################################################################
# Configuration
###############################################################################

DATA_DIR = "data/faces"     # change if your images live somewhere else
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["happy", "neutral", "sad"]
EMOJI_MAP = {"happy": "😊", "neutral": "😐", "sad": "😢"}


###############################################################################
# Reproducibility
###############################################################################

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


###############################################################################
# CNN Model
###############################################################################

class EmojifyCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


###############################################################################
# Data Loading
###############################################################################

def get_dataloaders(data_dir, batch_size, val_split):
    # Base transforms (used by both)
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # We first create a dataset without transform to generate indices
    base_dataset = datasets.ImageFolder(root=data_dir)
    print("Class-to-index mapping:", base_dataset.class_to_idx)

    dataset_size = len(base_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    # Deterministic split
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        base_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Now re-wrap subsets with transforms
    # Each Subset has .dataset and .indices; we create new ImageFolders with transforms
    full_train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    full_val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_dataset.indices)
    val_dataset = torch.utils.data.Subset(full_val_dataset, val_dataset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader, base_dataset.classes


###############################################################################
# Training / Evaluation
###############################################################################

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    correct, total = 0, 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # New: progress print
        if (batch_idx + 1) % 50 == 0:
            print(f"  [batch {batch_idx+1}] loss: {epoch_loss/total:.4f}")

    return epoch_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "loss": epoch_loss / len(all_labels),
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(len(class_names)), class_names)
    ax.set_yticks(np.arange(len(class_names)), class_names)
    plt.show()


###############################################################################
# Inference
###############################################################################

def load_trained_model(model_path, num_classes, device):
    model = EmojifyCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)


###############################################################################
# Main
###############################################################################

def main():

    train_acc_hist = []
    val_acc_hist = []
    val_f1_hist = []
    train_loss_hist = []
    val_loss_hist = []

    set_seed(RANDOM_SEED)

    print(f"Using device: {DEVICE}")

    train_loader, val_loader, class_names = get_dataloaders(DATA_DIR, BATCH_SIZE, VAL_SPLIT)
    print("Classes detected:", class_names)

    model = EmojifyCNN(num_classes=len(class_names)).to(DEVICE)

    # class order is ['happy', 'neutral', 'sad']
    class_weights = torch.tensor([0.9, 1.1, 1.2], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0
    best_model_path = "best_emojify_cnn.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f"Val loss: {val_metrics['loss']:.4f}, Val acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}")

        # 🔹 save best model
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model → {best_model_path}")

        # 🔹 record metrics for plotting
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_metrics["accuracy"])
        val_f1_hist.append(val_metrics["f1_macro"])
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_metrics["loss"])

    # Plot accuracy
    plt.figure()
    plt.plot(train_acc_hist, label="Train acc")
    plt.plot(val_acc_hist, label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot loss
    plt.figure()
    plt.plot(train_loss_hist, label="Train loss")
    plt.plot(val_loss_hist, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot F1
    plt.figure()
    plt.plot(val_f1_hist, label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 (macro)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nTraining complete!")



if __name__ == "__main__":
    main()
