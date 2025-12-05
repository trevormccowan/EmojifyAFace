import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = "best_emojify_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FACE_EMOJI_MAP = {
    0: "happy",
    1: "neutral",
    2: "sad",
}

# Use ASCII so OpenCV can draw it (no more ????)
ASCII_EMOJI_MAP = {
    "happy": ":)",
    "neutral": ":|",
    "sad": ":(",
}


# ---------------------------
# UPDATED MODEL - Must match training architecture
# ---------------------------
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
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------------
# UPDATED TRANSFORM - RGB, 224x224, ImageNet normalization
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
])


def predict_frame(model, frame_bgr, device):
    """
    Predict emotion from webcam frame.
    
    Args:
        model: trained CNN model
        frame_bgr: OpenCV frame (H, W, 3) in BGR format
        device: torch device
    
    Returns:
        emotion: string ("happy", "neutral", "sad")
        ascii_emoji: string (":)", ":|", ":(")
        confidence: float (0-1)
    """
    # frame_bgr: OpenCV frame (H, W, 3) in BGR
    h, w, _ = frame_bgr.shape
    
    # Center square crop to roughly isolate face
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame_bgr[y0:y0 + side, x0:x0 + side]

    # Convert BGR -> RGB (keep as RGB, no grayscale!)
    frame_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    # Apply same transform as validation
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)
        idx = pred.item()
        confidence = probabilities[0][idx].item()

    emotion = FACE_EMOJI_MAP[idx]
    ascii_emoji = ASCII_EMOJI_MAP[emotion]
    return emotion, ascii_emoji, confidence


def main():
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    model = load_model(MODEL_PATH, DEVICE)
    print("Model loaded successfully!")

    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully!")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            emotion, emoji_ascii, confidence = predict_frame(model, frame, DEVICE)
            label_text = f"{emotion} {emoji_ascii} ({confidence*100:.1f}%)"
        except Exception as e:
            label_text = f"Error: {str(e)[:20]}"

        # Draw label on the frame (top-left corner)
        cv2.putText(
            frame,
            label_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Emojify Webcam", frame)

        # quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    main()