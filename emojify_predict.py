import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse


MODEL_PATH = "best_emojify_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FACE_EMOJI_MAP = {
    0: "happy",
    1: "neutral",
    2: "sad",
}

UNICODE_EMOJI_MAP = {
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
}


class EmojifyCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
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


def load_model(model_path, device):
    model = EmojifyCNN(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def predict(model, image_path, device):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
        class_idx = pred.item()

    emotion = FACE_EMOJI_MAP[class_idx]
    unicode_emoji = UNICODE_EMOJI_MAP[emotion]
    return emotion, unicode_emoji


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to image file")
    args = parser.parse_args()

    model = load_model(MODEL_PATH, DEVICE)
    emotion, emoji = predict(model, args.image_path, DEVICE)

    print("\n===== RESULT =====")
    print(f"Predicted emotion: {emotion}")
    print(f"Unicode emoji:     {emoji}")
    print("==================\n")


if __name__ == "__main__":
    main()
