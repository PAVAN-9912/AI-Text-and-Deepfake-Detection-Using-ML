import os
import sys
import time
import cv2
import torch
import torchvision.models as models
import numpy as np
import pandas as pd

print("=" * 75)
print("EXTRACTING FACE-CROPPED TEMPORAL RESNET-18 FEATURES")
print("===================================================")

BASE_DIR = r"C:\Users\PAVAN S\OneDrive\Desktop\AI_Content_Detection"
DATA_DIR = os.path.join(BASE_DIR, "datasets", "video")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
META_PATH = os.path.join(FRAMES_DIR, "frame_metadata.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "face_temporal_features.csv")
CHECKPOINT_CSV = os.path.join(DATA_DIR, "face_temporal_checkpoint.csv")
ONNX_PATH = os.path.join(BASE_DIR, "models", "face_detector", "face_detection_yunet.onnx")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Initialize YuNet Face Detector
detector = cv2.FaceDetectorYN_create(
    model=ONNX_PATH,
    config="",
    input_size=(224, 224),
    score_threshold=0.5,
    nms_threshold=0.3,
    top_k=10
)

def extract_face_crop(frame, target_size=(224, 224), margin=0.25):
    h, w, _ = frame.shape
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is not None and len(faces) > 0:
        best_face = max(faces, key=lambda f: f[-1])
        x, y, fw, fh = best_face[0:4].astype(int)
        mx, my = int(fw * margin), int(fh * margin)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(w, x + fw + mx), min(h, y + fh + my)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] > 10 and crop.shape[1] > 10:
            return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA), True
    min_dim = min(h, w)
    cy, cx = h // 2, w // 2
    crop = frame[max(0, cy - min_dim // 2):max(0, cy - min_dim // 2) + min_dim, max(0, cx - min_dim // 2):max(0, cx - min_dim // 2) + min_dim]
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA), False

# Load ResNet-18 Backbone
weights = models.ResNet18_Weights.DEFAULT
resnet = models.resnet18(weights=weights)
resnet.fc = torch.nn.Identity()
resnet.to(DEVICE)
resnet.eval()
transforms = weights.transforms()

# Read Metadata
metadata = pd.read_csv(META_PATH)
total_videos = len(metadata)
print(f"Total videos to process: {total_videos}")

# Check for existing checkpoint
processed_ids = set()
existing_rows = []
if os.path.exists(CHECKPOINT_CSV):
    try:
        chk_df = pd.read_csv(CHECKPOINT_CSV)
        processed_ids = set(chk_df["video_id"])
        existing_rows = chk_df.to_dict(orient="records")
        print(f"Resuming from checkpoint: {len(processed_ids)} videos already processed.")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")

t0 = time.time()
rows_accumulated = existing_rows
checkpoint_interval = 200

for idx, row in metadata.iterrows():
    vid_id = row["video_id"]
    if vid_id in processed_ids:
        continue

    split = row["split"]
    label_name = row["label_name"]
    fdir = os.path.join(FRAMES_DIR, split, label_name, str(vid_id))

    if not os.path.exists(fdir):
        continue

    frame_files = sorted([os.path.join(fdir, f) for f in os.listdir(fdir) if f.endswith(".jpg")])
    if len(frame_files) == 0:
        continue

    # Sample exactly 8 frames
    if len(frame_files) >= 8:
        indices = np.linspace(0, len(frame_files) - 1, 8, dtype=int)
        sampled_files = [frame_files[i] for i in indices]
    else:
        sampled_files = frame_files + [frame_files[-1]] * (8 - len(frame_files))

    tensor_list = []
    face_detected_count = 0
    for fp in sampled_files:
        img = cv2.imread(fp)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        crop, detected = extract_face_crop(img)
        if detected:
            face_detected_count += 1
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = transforms(torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0)
        tensor_list.append(tensor)

    batch_tensors = torch.stack(tensor_list).to(DEVICE)
    with torch.no_grad():
        feats = resnet(batch_tensors).cpu().numpy().astype(np.float32) # (8, 512)

    # Flatten into row dict
    row_dict = {
        "split": split,
        "label": int(row["label"]),
        "label_name": label_name,
        "video_id": vid_id,
        "filename": row["filename"],
        "frame_count": 8,
        "face_detected_frames": face_detected_count,
        "frame_directory": fdir
    }
    for f_idx in range(8):
        for d_idx in range(512):
            row_dict[f"frame_{f_idx+1}_feature_{d_idx}"] = float(feats[f_idx, d_idx])

    rows_accumulated.append(row_dict)
    processed_ids.add(vid_id)

    if len(rows_accumulated) % checkpoint_interval == 0 or len(rows_accumulated) == total_videos:
        pd.DataFrame(rows_accumulated).to_csv(CHECKPOINT_CSV, index=False)
        elapsed = time.time() - t0
        speed = elapsed / max(len(processed_ids) - len(existing_rows), 1)
        remaining = (total_videos - len(processed_ids)) * speed
        print(f"Progress: {len(processed_ids)}/{total_videos} ({len(processed_ids)/total_videos*100:.1f}%) | Elapsed: {elapsed/60:.1f}m | Est. Remaining: {remaining/60:.1f}m")

# Final Save
df_final = pd.DataFrame(rows_accumulated)
df_final.to_csv(OUTPUT_CSV, index=False)
if os.path.exists(CHECKPOINT_CSV):
    try:
        os.remove(CHECKPOINT_CSV)
    except Exception:
        pass

print(f"\nEXTRACTION COMPLETE! Saved {len(df_final)} rows to {OUTPUT_CSV}")
print(f"Total time: {(time.time() - t0)/60:.2f} minutes")
