import sys
import os
import platform
import psutil
import torch
import shutil

print("="*70)
print("1. PYTHON & ENVIRONMENT")
print("="*70)
print(f"Python Version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")

print("\n" + "="*70)
print("2. PYTORCH & ACCELERATION (CUDA / CPU)")
print("="*70)
print(f"PyTorch Version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")
if cuda_available:
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"Total VRAM: {props.total_memory / (1024**3):.2f} GB")
    print(f"Allocated VRAM: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
    print(f"Cached VRAM: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
else:
    print("CUDA is NOT available. PyTorch will execute on CPU.")

print("\n" + "="*70)
print("3. CPU & SYSTEM RAM")
print("="*70)
print(f"CPU Model / Processor: {platform.processor()}")
print(f"Physical CPU Cores: {psutil.cpu_count(logical=False)}")
print(f"Logical CPU Cores: {psutil.cpu_count(logical=True)}")
mem = psutil.virtual_memory()
print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
print(f"Available RAM: {mem.available / (1024**3):.2f} GB ({mem.percent}% used)")

print("\n" + "="*70)
print("4. DISK SPACE")
print("="*70)
total_disk, used_disk, free_disk = shutil.disk_usage(".")
print(f"Current Drive Total Disk: {total_disk / (1024**3):.2f} GB")
print(f"Used Disk: {used_disk / (1024**3):.2f} GB")
print(f"Free Disk Space: {free_disk / (1024**3):.2f} GB")

print("\n" + "="*70)
print("5. TRANSFORMER & ML PACKAGES")
print("="*70)
try:
    import transformers
    print(f"Transformers Version: {transformers.__version__}")
except ImportError as e:
    print(f"Transformers: Not installed ({e})")

try:
    import tokenizers
    print(f"Tokenizers Version: {tokenizers.__version__}")
except ImportError as e:
    print(f"Tokenizers: Not installed ({e})")

try:
    import sentence_transformers
    print(f"Sentence-Transformers: Installed ({sentence_transformers.__version__})")
except Exception as e:
    print(f"Sentence-Transformers: Not directly loadable ({e})")

try:
    import sklearn
    print(f"Scikit-Learn Version: {sklearn.__version__}")
except ImportError as e:
    print(f"Scikit-Learn: Not installed ({e})")

print("\n" + "="*70)
print("6. HUGGING FACE CACHE CHECK")
print("="*70)
hf_cache = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface/hub"))
print(f"HF Cache Location: {hf_cache}")
if os.path.exists(hf_cache):
    print("Cached Models/Snapshots in HF Hub:")
    for root, dirs, files in os.walk(hf_cache):
        # List top level model folders
        if root == hf_cache:
            for d in dirs:
                print(f"  - {d}")
else:
    print("HF Cache directory does not exist or is empty.")
