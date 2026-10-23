import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.predict_video_vit import predict_video_vit, load_vit_artifacts

load_vit_artifacts()

dl = Path(r'C:\Users\PAVAN S\Downloads')
test_files = [
    ('Comedy (Real)', next(dl.glob('*Wholesome*.mp4'), None), 'REAL'),
    ('Kannada Comedy (Real)', next(dl.glob('*kannada*.mp4'), None), 'REAL'),
    ('WhatsApp (Real)', next(dl.glob('*WhatsApp Video 2026-08-18*.mp4'), None), 'REAL'),
    ('Hayabusa (Real No-Face)', next(dl.glob('*HAYABUSA*.mp4'), None), 'REAL'),
    ('Fake All Along (Fake Swap)', next(dl.glob('*fake all along*.mp4'), None), 'FAKE'),
    ('Thithi Rap (Fake)', next(dl.glob('*THITHI*.mp4'), None), 'FAKE'),
    ('AI Video Workflow (Gen AI)', next(dl.glob('*30 Minute*.mp4'), None), 'FAKE'),
    ('AI Baby Golgappe (Diffusion)', next(dl.glob('*Golgappe*.mp4'), None), 'FAKE'),
]

header = f"{'Video Description':<30} | {'True':<6} | {'ViT Pred':<8} | {'P(Real)':<8} | {'P(Fake)':<8} | {'Faces':<5} | {'Rel':<8} | {'Correct':<7}"
print(header)
print('-' * len(header))

for desc, path, true_label in test_files:
    if not path or not path.exists():
        print(f"{desc:<30} | NOT FOUND")
        continue
    res = predict_video_vit(str(path))
    pred_label = res['label']
    correct = (pred_label == true_label)
    correct_str = "YES" if correct else "NO"
    print(f"{desc:<30} | {true_label:<6} | {pred_label:8s} | {res['real_probability']:8.4f} | {res['fake_probability']:8.4f} | {res['face_count']:5d} | {res['reliability']:8s} | {correct_str:<7}")
