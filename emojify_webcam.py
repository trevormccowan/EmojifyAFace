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
# Model (same as training)
# ---------------------------
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


def load_model(model_path, device):
    model = EmojifyCNN(num_classes=3)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# same preprocessing as training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def predict_frame(model, frame_bgr, device):
    # frame_bgr: OpenCV frame (H, W, 3) in BGR
    h, w, _ = frame_bgr.shape
    # center square crop to roughly isolate face
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame_bgr[y0:y0 + side, x0:x0 + side]

    # convert BGR -> RGB and to PIL
    frame_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
        idx = pred.item()

    emotion = FACE_EMOJI_MAP[idx]
    ascii_emoji = ASCII_EMOJI_MAP[emotion]
    return emotion, ascii_emoji


def main():
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    model = load_model(MODEL_PATH, DEVICE)

    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            emotion, emoji_ascii = predict_frame(model, frame, DEVICE)
            label_text = f"{emotion} {emoji_ascii}"
        except Exception:
            label_text = "unknown :("

        # Draw label on the frame (top-left corner)
        cv2.putText(
            frame,
            label_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
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


if __name__ == "__main__":
    main()
