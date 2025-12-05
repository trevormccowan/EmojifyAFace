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


###############################################################################
# UPDATED MODEL - Must match training architecture
###############################################################################

class EmojifyCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Block 1: 3 -> 32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Block 2: 32 -> 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Block 3: 64 -> 128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Block 4: 128 -> 256
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Adaptive pooling to handle any input size
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        # Classifier
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_model(model_path, device):
    model = EmojifyCNN(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


###############################################################################
# UPDATED TRANSFORM - Must match training (RGB, 224x224, ImageNet norm)
###############################################################################

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
])


def predict(model, image_path, device):
    # Load image as RGB (no conversion to grayscale!)
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)
        class_idx = pred.item()
        confidence = probabilities[0][class_idx].item()

    emotion = FACE_EMOJI_MAP[class_idx]
    unicode_emoji = UNICODE_EMOJI_MAP[emotion]
    return emotion, unicode_emoji, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to image file")
    args = parser.parse_args()

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, DEVICE)
    print(f"Model loaded successfully on {DEVICE}")
    
    print(f"Predicting emotion for: {args.image_path}")
    emotion, emoji, confidence = predict(model, args.image_path, DEVICE)

    print("\n===== RESULT =====")
    print(f"Predicted emotion: {emotion}")
    print(f"Unicode emoji:     {emoji}")
    print(f"Confidence:        {confidence*100:.1f}%")
    print("==================\n")


if __name__ == "__main__":
    main()