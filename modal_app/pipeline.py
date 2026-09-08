"""Modal Pipeline Orchestrator
Điều phối thực hiện toàn bộ quy trình với cơ chế bảo toàn dữ liệu khi Timeout / Dừng đột ngột:
1. Download Dataset từ Roboflow vào Modal Volume (/vol/datasets)
2. Train RT-DETR 50 epochs trên Modal GPU (Auto-save / Auto-resume)
3. Train RF-DETR 50 epochs trên Modal GPU (Auto-save / Auto-resume)
4. Evaluate RT-DETR và RF-DETR trên tập test
5. Sinh báo cáo so sánh tổng thể
"""
import os
import sys
import modal

# 1. Định nghĩa Modal App
app = modal.App("food-detection-training")

# 2. Định nghĩa Modal Volume lưu trữ bền vững
volume = modal.Volume.from_name("food-detection-training", create_if_missing=True)

# 3. Định nghĩa Modal Secret cho Roboflow
secret = modal.Secret.from_name("roboflow-secret")

# 4. Định nghĩa Container Image
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
        "scipy>=1.12.0",
        "accelerate>=0.28.0",
        "pytorch-lightning>=2.2.0",
        "einops>=0.7.0",
        "supervision>=0.19.0",
        "tensorboard",
        "rfdetr[train,loggers]",
    )
    .env({
        "PYTHONPATH": "/root:/root/src:.",
        "YOLO_CONFIG_DIR": "/tmp/Ultralytics",
    })
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(
    image=image,
    volumes={"/vol": volume},
    secrets=[secret] if secret else [],
    timeout=1800,  # 30 phút
)
def download_dataset_step(roboflow_api_key: str = None, version: int = 5, force: bool = False):
    import shutil
    from src.data.roboflow_download import download_dataset
    from src.data.validate_dataset import validate_yolo_dataset

    volume.reload()
    
    # 1. Dọn dẹp dataset v1 cũ theo yêu cầu để giải phóng bộ nhớ
    old_v1_dir = "/vol/datasets/completed-project-1"
    if os.path.exists(old_v1_dir):
        print(f"[*] Đang xóa bỏ dataset v1 cũ: {old_v1_dir}...")
        shutil.rmtree(old_v1_dir, ignore_errors=True)
        print(f"[+] Đã dọn dẹp sạch sẽ {old_v1_dir}")

    print(f"[*] STEP 1: Kiểm tra & Tải Roboflow Dataset (Version {version}, Force={force})...")
    res = download_dataset(
        api_key=roboflow_api_key,
        version_num=version,
        force_redownload=force,
    )
    volume.commit()
    print("[+] Dataset v5 đã sẵn sàng và được commit vào volume.")

    # Validate dataset
    yolo_yaml = os.path.join(res["yolo_path"], "data.yaml")
    if os.path.exists(yolo_yaml):
        stats = validate_yolo_dataset(yolo_yaml)
        print("[+] Dataset Validation thành công!")
    return res


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=300,
)
def clean_volume_state():
    """Dọn dẹp sạch sẽ các checkpoints, logs, outputs cũ để bắt đầu huấn luyện mới 100% từ Epoch 1."""
    import shutil
    volume.reload()
    print("[*] Đang tiến hành dọn dẹp sạch sẽ checkpoints, logs và outputs cũ trên Volume...")
    
    for path in ["/vol/checkpoints", "/vol/logs", "/vol/outputs"]:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
            print(f"[+] Đã xóa: {path}")
    
    # Tạo lại các thư mục sạch sẽ
    os.makedirs("/vol/checkpoints/rtdetr", exist_ok=True)
    os.makedirs("/vol/checkpoints/rfdetr", exist_ok=True)
    os.makedirs("/vol/logs/rtdetr", exist_ok=True)
    os.makedirs("/vol/logs/rfdetr", exist_ok=True)
    os.makedirs("/vol/outputs/rtdetr_eval", exist_ok=True)
    os.makedirs("/vol/outputs/rfdetr_eval", exist_ok=True)
    
    volume.commit()
    print("[+] ĐÃ HOÀN TẤT LÀM SẠCH STORAGE! Trạng thái sẵn sàng 100% cho Dataset v5 từ Epoch 1.")
    return True


@app.local_entrypoint()
def reset_runs():
    """Lệnh dọn dẹp sạch sẽ toàn bộ trạng thái chạy cũ (checkpoints, logs, outputs)."""
    clean_volume_state.remote()


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": volume},
    timeout=43200,  # 12 giờ (thoải mái cho 50 epochs)
)
def train_rtdetr_step():
    from src.rtdetr.train import train_rtdetr

    volume.reload()
    print("[*] STEP 2: Training RT-DETR for 50 epochs on GPU A100...")
    try:
        best_ckpt = train_rtdetr(
            data_yaml_path="/vol/datasets/completed-project-5/yolo/data.yaml",
            config_path="/root/configs/rtdetr.yaml",
            output_dir="/vol/checkpoints/rtdetr",
            log_dir="/vol/logs/rtdetr",
            commit_fn=volume.commit,
        )
    finally:
        # Luôn commit volume bất kể thành công hay bị timeout
        volume.commit()

    return best_ckpt


def _ensure_coco_dataset_ready():
    import os
    import shutil
    from pathlib import Path
    from roboflow import Roboflow

    coco_dir = "/vol/datasets/completed-project-5/coco"
    coco_test = "/vol/datasets/completed-project-5/coco_test"

    if os.path.exists(coco_test):
        shutil.rmtree(coco_test, ignore_errors=True)

    train_json = Path(coco_dir) / "train" / "_annotations.coco.json"
    if not train_json.exists():
        print(f"[*] COCO dataset chưa sẵn sàng hoặc thiếu file JSON. Đang tải định dạng COCO từ Roboflow vào {coco_dir}...")
        if os.path.exists(coco_dir):
            shutil.rmtree(coco_dir, ignore_errors=True)
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("nckhcict2025").project("completed-project")
        version = project.version(5)
        version.download("coco", location=coco_dir)
        print(f"[+] Tải COCO thành công! Kiểm tra train json: {train_json.exists()}")

    stray_yaml = Path(coco_dir) / "data.yaml"
    if stray_yaml.exists():
        try:
            stray_yaml.unlink()
        except Exception:
            pass

    volume.commit()
    return coco_dir


@app.function(
    image=image,
    volumes={"/vol": volume},
    secrets=[secret] if secret else [],
    timeout=1800,
)
def setup_coco_step():
    """Tải và chuẩn bị sẵn sàng tập dữ liệu chuẩn COCO JSON trên CPU (tiết kiệm GPU credits)."""
    volume.reload()
    coco_dir = _ensure_coco_dataset_ready()
    volume.commit()
    print(f"[+] Tập dữ liệu COCO đã sẵn sàng tại: {coco_dir}")
    return True


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": volume},
    secrets=[secret] if secret else [],
    timeout=43200,  # 12 giờ
)
def train_rfdetr_step():
    from src.rfdetr.train import train_rfdetr

    volume.reload()
    print("[*] STEP 3: Training RF-DETR for 50 epochs on GPU A100...")
    coco_dir = _ensure_coco_dataset_ready()
    try:
        best_ckpt = train_rfdetr(
            dataset_dir=coco_dir,
            config_path="/root/configs/rfdetr.yaml",
            output_dir="/vol/checkpoints/rfdetr",
            log_dir="/vol/logs/rfdetr",
            commit_fn=volume.commit,
        )
    finally:
        # Luôn commit volume bất kể thành công hay bị timeout
        volume.commit()

    return best_ckpt


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": volume},
    timeout=3600,
)
def evaluate_both_models():
    from src.rtdetr.evaluate import evaluate_rtdetr
    from src.rfdetr.evaluate import evaluate_rfdetr
    from src.common.metrics_reporter import MetricsReporter

    volume.reload()
    print("[*] STEP 4: Evaluating RT-DETR and RF-DETR on Test split...")
    
    # 1. RT-DETR eval (YOLO format)
    rt_metrics, rt_report = evaluate_rtdetr(
        checkpoint_path="/vol/checkpoints/rtdetr/best.pt",
        data_yaml_path="/vol/datasets/completed-project-5/yolo/data.yaml",
        output_dir="/vol/outputs/rtdetr_eval",
    )

    # 2. RF-DETR eval (COCO format)
    rf_metrics, rf_report = evaluate_rfdetr(
        checkpoint_path="/vol/checkpoints/rfdetr/best.pth",
        dataset_dir="/vol/datasets/completed-project-5/coco",
        output_dir="/vol/outputs/rfdetr_eval",
    )

    # 3. Generate comparison report
    comp_file = "/vol/logs/comparison_report.md"
    rt_json = "/vol/outputs/rtdetr_eval/rt-detr_test_report.json"
    rf_json = "/vol/outputs/rfdetr_eval/rf-detr_test_report.json"

    if os.path.exists(rt_json) and os.path.exists(rf_json):
        MetricsReporter.generate_comparison_report(rt_json, rf_json, comp_file)
        print(f"[+] Comparison report generated at: {comp_file}")

    volume.commit()
    return {"rt_metrics": rt_metrics, "rf_metrics": rf_metrics, "comparison_report": comp_file}


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": volume},
    timeout=600,
)
def evaluate_rtdetr_step():
    from src.rtdetr.evaluate import evaluate_rtdetr
    volume.reload()
    rt_metrics, rt_report = evaluate_rtdetr(
        checkpoint_path="/vol/checkpoints/rtdetr/best.pt",
        data_yaml_path="/vol/datasets/completed-project-5/yolo/data.yaml",
        output_dir="/vol/outputs/rtdetr_eval",
    )
    volume.commit()
    return rt_metrics, rt_report


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": volume},
    timeout=600,
)
def evaluate_rfdetr_step():
    from src.rfdetr.evaluate import evaluate_rfdetr
    volume.reload()
    rf_metrics, rf_report = evaluate_rfdetr(
        checkpoint_path="/vol/checkpoints/rfdetr/best.pth",
        dataset_dir="/vol/datasets/completed-project-5/coco",
        output_dir="/vol/outputs/rfdetr_eval",
    )
    volume.commit()
    return rf_metrics, rf_report


@app.local_entrypoint()
def eval_rtdetr():
    """Chạy đánh giá RT-DETR trên tập test."""
    metrics, report = evaluate_rtdetr_step.remote()
    print("[+] Hoàn tất đánh giá RT-DETR trên Test Split!")
    print(metrics)


@app.local_entrypoint()
def eval_rfdetr():
    """Chạy đánh giá RF-DETR trên tập test."""
    metrics, report = evaluate_rfdetr_step.remote()
    print("[+] Hoàn tất đánh giá RF-DETR trên Test Split!")
    print(metrics)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": volume},
    secrets=[secret] if secret else [],
    timeout=43200,  # 12 giờ
)
def run_rfdetr_cloud_pipeline():
    """Hàm chạy 100% trên Cloud: Huấn luyện RF-DETR nốt 20 epochs và tự động xuất báo cáo so sánh."""
    from src.rfdetr.train import train_rfdetr
    from src.rtdetr.evaluate import evaluate_rtdetr
    from src.rfdetr.evaluate import evaluate_rfdetr
    from src.common.metrics_reporter import MetricsReporter
    
    volume.reload()
    coco_dir = _ensure_coco_dataset_ready()
    print("[*] [CLOUD TASK] Bắt đầu tiếp tục huấn luyện RF-DETR trên GPU A100...")
    rf_ckpt = train_rfdetr(
        dataset_dir=coco_dir,
        config_path="configs/rfdetr.yaml",
        output_dir="/vol/checkpoints/rfdetr",
        log_dir="/vol/logs/rfdetr",
        commit_fn=volume.commit,
    )
    volume.commit()
    print(f"[+] [CLOUD TASK] RF-DETR đã hoàn thành! Checkpoint: {rf_ckpt}")

    print("\n[*] [CLOUD TASK] Đang đánh giá cả 2 model trên Test Split...")
    output_dir = "/vol/outputs"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/rtdetr_eval", exist_ok=True)
    os.makedirs(f"{output_dir}/rfdetr_eval", exist_ok=True)

    rt_metrics, rt_report = evaluate_rtdetr(
        checkpoint_path="/vol/checkpoints/rtdetr/best.pt",
        data_yaml_path="/vol/datasets/completed-project-5/yolo/data.yaml",
        output_dir=f"{output_dir}/rtdetr_eval",
    )
    rf_metrics, rf_report = evaluate_rfdetr(
        checkpoint_path=rf_ckpt,
        dataset_dir=coco_dir,
        output_dir=f"{output_dir}/rfdetr_eval",
    )

    rt_json = f"{output_dir}/rtdetr_eval/rt-detr_test_report.json"
    rf_json = f"{output_dir}/rfdetr_eval/rf-detr_test_report.json"
    comp_file = f"{output_dir}/comparison_report.md"
    if os.path.exists(rt_json) and os.path.exists(rf_json):
        MetricsReporter.generate_comparison_report(rt_json, rf_json, comp_file)
        print(f"[+] [CLOUD TASK] Báo cáo so sánh đối đầu đã tạo tại: {comp_file}")

    volume.commit()
    print("[+] [CLOUD TASK] HOÀN TẤT TOÀN BỘ PIPELINE TRÊN CLOUD!")
    return {"rf_checkpoint": rf_ckpt, "comparison_report": comp_file}


@app.local_entrypoint()
def eval_all():
    """Chạy đánh giá cả RT-DETR và RF-DETR trên tập test và xuất báo cáo so sánh."""
    print("==================================================================")
    print("  ĐÁNH GIÁ CẢ 2 MÔ HÌNH TRÊN TEST SPLIT & TẠO BÁO CÁO SO SÁNH")
    print("==================================================================")
    results = evaluate_both_models.remote()
    print("\n[+] HOÀN TẤT ĐÁNH GIÁ VÀ XUẤT BÁO CÁO SO SÁNH!")
    print(results)


@app.local_entrypoint()
def resume_rfdetr():
    """Kích hoạt RF-DETR chạy ngầm 100% trên Cloud Modal (Tắt terminal thoải mái)."""
    print("==================================================================")
    print("  TIẾP TỤC HUẤN LUYỆN RF-DETR TRÊN CLOUD (TẮT TERMINAL THOẢI MÁI)")
    print("==================================================================")
    print("\n[*] Đang gửi tác vụ lên cụm máy chủ Cloud Modal GPU A100...")
    call = run_rfdetr_cloud_pipeline.spawn()
    print(f"\n[+] ĐÃ KÍCH HOẠT THÀNH CÔNG LÊN CLOUD!")
    print(f"[*] Function Call ID: {call.object_id}")
    print(f"[*] Tiến trình hiện đang chạy 100% trên GPU A100 của Cloud Modal.")
    print(f"[*] BÂY GIỜ BẠN CÓ THỂ TẮT HOÀN TOÀN TERMINAL VÀ TẮT MÁY MÀ KHÔNG BỊ DỪNG!")
    print(f"[*] Xem log trực tiếp tại: https://modal.com/apps")


@app.local_entrypoint()
def train_parallel():
    """Chạy trực tiếp huấn luyện song song 2 mô hình trên 2 GPU A100 (khi dataset đã có sẵn)."""
    print("==================================================================")
    print("  HUẤN LUYỆN SONG SONG: RT-DETR & RF-DETR TRÊN 2 GPU NVIDIA A100")
    print("==================================================================")
    
    print("\n[*] Kích hoạt đồng thời 2 GPU A100...")
    rt_future = train_rtdetr_step.spawn()
    rf_future = train_rfdetr_step.spawn()

    print("[*] Đang chờ 2 mô hình huấn luyện song song hoàn tất...")
    rt_ckpt = rt_future.get()
    print(f"[+] RT-DETR hoàn thành! Checkpoint: {rt_ckpt}")

    rf_ckpt = rf_future.get()
    print(f"[+] RF-DETR hoàn thành! Checkpoint: {rf_ckpt}")

    print("\n--- Đánh Giá & Xuất Báo Cáo So Sánh ---")
    results = evaluate_both_models.remote()
    print("\n[+] HOÀN TẤT TOÀN BỘ!")
    print(results)


@app.local_entrypoint()
def main(roboflow_key: str = "", skip_download: bool = False, version: int = 5):
    print("==================================================================")
    print("  PIPELINE HUẤN LUYỆN SONG SONG: RT-DETR & RF-DETR TRÊN MODAL")
    print("==================================================================")
    
    # 1. Download Dataset nếu chưa có
    if not skip_download:
        print(f"\n--- [BƯỚC 1] Kiểm tra / Tải Dataset (v{version}) vào Modal Volume ---")
        download_dataset_step.remote(roboflow_api_key=roboflow_key, version=version)
    else:
        print("\n--- [BƯỚC 1] Bỏ qua tải dataset, sử dụng dữ liệu có sẵn trong Volume ---")

    # 2. Huấn luyện SONG SONG 2 GPU A100
    print("\n--- [BƯỚC 2] Huấn Luyện Song Song Cả 2 Model Trên 2 GPU A100 ---")
    rt_future = train_rtdetr_step.spawn()
    rf_future = train_rfdetr_step.spawn()

    print("[*] Đang chạy song song 2 GPU A100...")
    rt_ckpt = rt_future.get()
    print(f"[+] RT-DETR hoàn tất! Checkpoint: {rt_ckpt}")

    rf_ckpt = rf_future.get()
    print(f"[+] RF-DETR hoàn tất! Checkpoint: {rf_ckpt}")

    # 3. Evaluation & Comparison
    print("\n--- [BƯỚC 3] Đánh giá & Xuất Báo Cáo So Sánh ---")
    results = evaluate_both_models.remote()
    print("\n[+] HOÀN THÀNH TOÀN BỘ PIPELINE!")
    print(results)
