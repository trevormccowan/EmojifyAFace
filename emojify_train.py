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

# Map class indices -> labels -> emojis
CLASS_NAMES = ["happy", "neutral", "sad"]  # must match your folder names order
EMOJI_MAP = {
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
}


###############################################################################
# Reproducibility helpers
###############################################################################

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###############################################################################
# CNN Model
###############################################################################

class EmojifyCNN(nn.Module):
    """
    Simple CNN for 128x128 grayscale images with 3 emotion classes.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Input: (1, 128, 128)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # (32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # (64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # (128, 16, 16)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


###############################################################################
# Data loading
###############################################################################

def get_dataloaders(data_dir: str, batch_size: int, val_split: float):
    """
    Uses ImageFolder to load all images, then splits into train/val.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # Normalize to roughly zero mean and unit variance
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Optional: print to verify class -> index mapping
    print("Class-to-index mapping from ImageFolder:", full_dataset.class_to_idx)

    # Ensure the order of CLASS_NAMES matches this mapping if you care
    # about consistent interpretation later.

    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, full_dataset.classes  # classes list


###############################################################################
# Training and evaluation loops
###############################################################################

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = epoch_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = epoch_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm,
    }
    return metrics


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Print values in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.show()


###############################################################################
# Inference helper: image -> emoji
###############################################################################

def load_trained_model(model_path: str, num_classes: int, device):
    model = EmojifyCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(model, image_path: str, class_names, device):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)  # shape: (1, 1, 128, 128)

    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
        pred_idx = pred.item()

    pred_label = class_names[pred_idx]
    emoji = EMOJI_MAP.get(pred_label, "?")
    return pred_label, emoji


###############################################################################
# Main
###############################################################################

def main():
    set_seed(RANDOM_SEED)

    print(f"Using device: {DEVICE}")

    # Data
    train_loader, val_loader, class_names = get_dataloaders(
        DATA_DIR, BATCH_SIZE, VAL_SPLIT
    )
    print("Classes detected:", class_names)

    # Model / loss / optimizer
    model = EmojifyCNN(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_f1 = 0.0
    best_model_path = "best_emojify_cnn.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(
            f"Val   loss: {val_metrics['loss']:.4f}, "
            f"Val acc: {val_metrics['accuracy']:.4f}, "
            f"Val F1 (macro): {val_metrics['f1_macro']:.4f}"
        )

        # Save best model by F1
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path}")

    # Final evaluation on validation set with best model
    print("\nLoading best model for final evaluation...")
    best_model = load_trained_model(best_model_path, num_classes=len(class_names), device=DEVICE)
    final_metrics = evaluate(best_model, val_loader, criterion, DEVICE)

    print("\nFinal validation metrics:")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision (macro): {final_metrics['precision_macro']:.4f}")
    print(f"Recall    (macro): {final_metrics['recall_macro']:.4f}")
    print(f"F1        (macro): {final_metrics['f1_macro']:.4f}")
    print("Confusion matrix:\n", final_metrics["confusion_matrix"])

    # Optional: plot confusion matrix
    plot_confusion_matrix(final_metrics["confusion_matrix"], class_names)

    # Example of using the emoji predictor
    example_image = None  # put path to a face image here, e.g. "test_face.png"
    if example_image is not None and os.path.exists(example_image):
        label, emoji = predict_image(best_model, example_image, class_names, DEVICE)
        print(f"Predicted label: {label}, emoji: {emoji}")


if __name__ == "__main__":
    main()
