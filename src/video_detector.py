import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = Path(
    r"models\video\temporal_attention_model.pth"
)

SCALER_FILE = Path(
    r"models\video\video_scaler.pkl"
)

CONFIG_FILE = Path(
    r"models\video\video_config.pkl"
)


SEQUENCE_LENGTH = 8
FEATURE_DIMENSION = 512

FRAME_WIDTH = 224
FRAME_HEIGHT = 224


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("VIDEO DEEPFAKE DETECTOR")
print("=======================")

print(
    "Device:",
    DEVICE
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_files = [
    MODEL_FILE,
    SCALER_FILE,
    CONFIG_FILE
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required model file not found: {file_path}"
        )


# ============================================================
# LOAD CONFIGURATION
# ============================================================

print(
    "\nLoading saved video model..."
)


with open(
    CONFIG_FILE,
    "rb"
) as file:

    config = pickle.load(
        file
    )


SEQUENCE_LENGTH = int(
    config.get(
        "sequence_length",
        SEQUENCE_LENGTH
    )
)


FEATURE_DIMENSION = int(
    config.get(
        "feature_dim",
        FEATURE_DIMENSION
    )
)


THRESHOLD = float(
    config.get(
        "threshold",
        0.50
    )
)


# ============================================================
# LOAD SCALER
# ============================================================

with open(
    SCALER_FILE,
    "rb"
) as file:

    scaler = pickle.load(
        file
    )


feature_mean = np.asarray(
    scaler["mean"],
    dtype=np.float32
)


feature_std = np.asarray(
    scaler["std"],
    dtype=np.float32
)


# ============================================================
# LOAD RESNET-18
# ============================================================

print(
    "Loading ResNet-18..."
)


weights = (
    ResNet18_Weights.DEFAULT
)


resnet = resnet18(
    weights=weights
)


# Remove ImageNet classifier.

resnet.fc = nn.Identity()


resnet = resnet.to(
    DEVICE
)


resnet.eval()


# IMPORTANT:
# Use exactly the same preprocessing as training.

transform = (
    weights.transforms()
)


# ============================================================
# TEMPORAL ATTENTION MODEL
# ============================================================

class TemporalAttentionModel(
    nn.Module
):

    def __init__(
        self,
        input_dim=512,
        hidden_dim=128
    ):

        super().__init__()


        # ----------------------------------------------------
        # 2-layer Bidirectional LSTM
        # ----------------------------------------------------

        self.lstm = nn.LSTM(

            input_size=input_dim,

            hidden_size=hidden_dim,

            num_layers=2,

            batch_first=True,

            dropout=0.2,

            bidirectional=True
        )


        attention_dim = (
            hidden_dim * 2
        )


        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.attention = nn.Sequential(

            nn.Linear(
                attention_dim,
                64
            ),

            nn.Tanh(),

            nn.Linear(
                64,
                1
            )
        )


        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                attention_dim,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                1
            )
        )


    def forward(
        self,
        x
    ):

        lstm_output, _ = self.lstm(
            x
        )


        attention_scores = (
            self.attention(
                lstm_output
            )
        )


        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )


        context = torch.sum(

            attention_weights
            * lstm_output,

            dim=1
        )


        logits = self.classifier(
            context
        ).squeeze(1)


        return logits


# ============================================================
# LOAD TEMPORAL MODEL
# ============================================================

model = TemporalAttentionModel(
    input_dim=FEATURE_DIMENSION,
    hidden_dim=128
)


state_dict = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)


model.load_state_dict(
    state_dict
)


model = model.to(
    DEVICE
)


model.eval()


print(
    "Temporal Attention model loaded."
)


print(
    "Threshold:",
    THRESHOLD
)


# ============================================================
# VIDEO FRAME EXTRACTION
# ============================================================

def extract_frames(
    video_path
):

    video_path = str(
        video_path
    )


    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        raise ValueError(
            "Could not open video: "
            + video_path
        )


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    if total_frames <= 0:

        cap.release()

        raise ValueError(
            "Video contains no readable frames."
        )


    # --------------------------------------------------------
    # Same frame-selection strategy used during training
    # --------------------------------------------------------

    if total_frames <= SEQUENCE_LENGTH:

        indices = np.arange(
            total_frames
        )

    else:

        indices = np.linspace(
            0,
            total_frames - 1,
            SEQUENCE_LENGTH,
            dtype=int
        )


    frames = []


    for frame_index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )


        success, frame = cap.read()


        if not success:

            continue


        # Same resizing used during
        # dataset frame extraction.

        frame = cv2.resize(

            frame,

            (
                FRAME_WIDTH,
                FRAME_HEIGHT
            ),

            interpolation=cv2.INTER_AREA
        )


        # OpenCV gives BGR.
        # Convert to RGB for PIL/ResNet.

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        frames.append(
            Image.fromarray(
                frame
            )
        )


    cap.release()


    if len(frames) == 0:

        raise ValueError(
            "Could not extract any frames."
        )


    # --------------------------------------------------------
    # Pad if fewer than 8 frames
    # --------------------------------------------------------

    while len(frames) < SEQUENCE_LENGTH:

        frames.append(
            frames[-1].copy()
        )


    # In case OpenCV unexpectedly returned
    # more frames than required.

    if len(frames) > SEQUENCE_LENGTH:

        indices = np.linspace(
            0,
            len(frames) - 1,
            SEQUENCE_LENGTH,
            dtype=int
        )


        frames = [
            frames[index]
            for index in indices
        ]


    return frames


# ============================================================
# EXTRACT RESNET FEATURES
# ============================================================

def extract_resnet_features(
    frames
):

    processed = []


    for frame in frames:

        tensor = transform(
            frame
        )


        processed.append(
            tensor
        )


    batch = torch.stack(
        processed
    ).to(
        DEVICE
    )


    with torch.no_grad():

        features = resnet(
            batch
        )


    features = (
        features
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


    expected_shape = (
        SEQUENCE_LENGTH,
        FEATURE_DIMENSION
    )


    if features.shape != expected_shape:

        raise ValueError(
            "Unexpected ResNet feature shape: "
            + str(features.shape)
        )


    return features


# ============================================================
# STANDARDIZE FEATURES
# ============================================================

def standardize_features(
    features
):

    return (
        features - feature_mean
    ) / feature_std


# ============================================================
# PREDICT VIDEO
# ============================================================

def predict_video(
    video_path
):

    print(
        "\nProcessing video..."
    )


    video_path = Path(
        video_path
    )


    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )


    if video_path.suffix.lower() not in [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    ]:

        raise ValueError(
            "Unsupported video format."
        )


    # --------------------------------------------------------
    # Extract 8 frames
    # --------------------------------------------------------

    frames = extract_frames(
        video_path
    )


    print(
        "Frames analyzed:",
        len(frames)
    )


    # --------------------------------------------------------
    # ResNet-18
    # --------------------------------------------------------

    features = extract_resnet_features(
        frames
    )


    print(
        "ResNet feature shape:",
        features.shape
    )


    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    features = standardize_features(
        features
    )


    # --------------------------------------------------------
    # Convert to PyTorch tensor
    # --------------------------------------------------------

    tensor = torch.tensor(
        features,
        dtype=torch.float32
    )


    # Add batch dimension:
    #
    # 8 × 512
    #
    # becomes
    #
    # 1 × 8 × 512

    tensor = tensor.unsqueeze(
        0
    ).to(
        DEVICE
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    model.eval()


    with torch.no_grad():

        logits = model(
            tensor
        )


        real_probability = (
            torch.sigmoid(
                logits
            )
            .item()
        )


    # IMPORTANT:
    # Training label mapping:
    #
    # 0 = Fake
    # 1 = Real

    fake_probability = (
        1.0 - real_probability
    )


    if real_probability >= THRESHOLD:

        prediction = "REAL"

        confidence = (
            real_probability
        )

    else:

        prediction = "FAKE"

        confidence = (
            fake_probability
        )


    return {

        "video": str(
            video_path
        ),

        "prediction":
            prediction,

        "confidence":
            confidence,

        "fake_probability":
            fake_probability,

        "real_probability":
            real_probability,

        "frames_analyzed":
            len(frames)
    }


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    result
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "VIDEO DETECTION RESULT"
    )

    print(
        "=" * 60
    )


    print(
        "\nVideo:",
        result["video"]
    )


    print(
        "\nFrames analyzed:",
        result["frames_analyzed"]
    )


    print(
        "\nPrediction:",
        result["prediction"]
    )


    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )


    print(
        "Fake probability:",
        f"{result['fake_probability'] * 100:.2f}%"
    )


    print(
        "Real probability:",
        f"{result['real_probability'] * 100:.2f}%"
    )


    print(
        "=" * 60
    )


# ============================================================
# MAIN INTERACTIVE LOOP
# ============================================================

if __name__ == "__main__":

    print(
        "\nVIDEO DETECTOR READY"
    )

    print(
        "===================="
    )

    print(
        "Enter the path of an MP4/video."
    )

    print(
        "Type 'exit' to stop."
    )


    while True:

        print(
            "\n"
            + "-" * 60
        )


        video_input = input(
            "Enter video path: "
        ).strip()


        if video_input.lower() == "exit":

            print(
                "\nExiting video detector."
            )

            break


        if not video_input:

            print(
                "Please enter a video path."
            )

            continue


        # Remove quotes if the user
        # pastes a quoted Windows path.

        video_input = (
            video_input
            .strip('"')
            .strip("'")
        )


        try:

            result = predict_video(
                video_input
            )


            display_result(
                result
            )


        except Exception as error:

            print(
                "\nERROR:"
            )

            print(
                error
            )