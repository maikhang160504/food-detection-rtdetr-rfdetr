"""Modal App Configuration
Thiết lập Modal App, GPU Container Image, Volume và Secret.
"""
import modal

# 1. Định nghĩa Modal App
app = modal.App("food-detection-training")

# 2. Định nghĩa Modal Volume lưu trữ bền vững
volume = modal.Volume.from_name("food-detection-training", create_if_missing=True)

# 3. Định nghĩa Modal Secret cho Roboflow
secret = modal.Secret.from_name("roboflow-secret")

# 4. Định nghĩa Container Image với đầy đủ CUDA, PyTorch, Ultralytics và RF-DETR
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "ultralytics>=8.3.0",
        "roboflow>=1.1.40",
        "pyyaml>=6.0.1",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "tabulate>=0.9.0",
        "tqdm>=4.66.0",
        "timm>=0.9.16",
        "pycocotools>=2.0.7",
        "scipy",
        "accelerate",
        "rfdetr[train,loggers]",
    )
    .env({"PYTHONPATH": "/root:/root/src"})
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("src", remote_path="/root/src")
)
