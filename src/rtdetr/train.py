"""RT-DETR Training Module
Huấn luyện mô hình RT-DETR 50 epochs với cơ chế bảo vệ checkpoint toàn diện khi hết giờ (Timeout / SIGTERM).
"""
import os
import signal
import sys
import shutil
import time
import yaml
from pathlib import Path
from typing import Optional, Callable
from ultralytics import RTDETR
from src.common.epoch_logger import EpochLogger


def train_rtdetr(
    data_yaml_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    config_path: str = "configs/rtdetr.yaml",
    output_dir: str = "/vol/checkpoints/rtdetr",
    log_dir: str = "/vol/logs/rtdetr",
    commit_fn: Optional[Callable[[], None]] = None,
    **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Chuẩn hóa data_yaml_path từ các alias
    if not data_yaml_path:
        if dataset_dir:
            data_yaml_path = dataset_dir if dataset_dir.endswith("data.yaml") else os.path.join(dataset_dir, "data.yaml")
        elif data_dir:
            data_yaml_path = data_dir if data_dir.endswith("data.yaml") else os.path.join(data_dir, "data.yaml")
        else:
            candidates = [
                "/vol/datasets/completed-project-5/yolo/data.yaml",
                "/vol/datasets/completed-project-1/yolo/data.yaml",
            ]
            for c in candidates:
                if os.path.exists(c):
                    data_yaml_path = c
                    break
            if not data_yaml_path:
                data_yaml_path = "/vol/datasets/completed-project-5/yolo/data.yaml"

    if os.path.isdir(data_yaml_path):
        data_yaml_path = os.path.join(data_yaml_path, "data.yaml")

    # Đọc cấu hình
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        alt_cfg = "/root/configs/rtdetr.yaml"
        if os.path.exists(alt_cfg):
            with open(alt_cfg, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    model_variant = model_cfg.get("variant", "rtdetr-l.pt")
    epochs = train_cfg.get("epochs", 80)
    imgsz = train_cfg.get("imgsz", 640)
    batch_size = train_cfg.get("batch_size", 16)
    lr0 = train_cfg.get("lr0", 0.0001)
    device = train_cfg.get("device", 0)

    run_name = f"run_{epochs}epochs"

    # 2. Kiểm tra checkpoint dở dang để tự động resume nếu trước đó bị hết giờ (Timeout)
    last_checkpoint = Path(output_dir) / "last.pt"
    run_last_checkpoint = Path(output_dir) / run_name / "weights" / "last.pt"
    legacy_run_last = Path(output_dir) / "run_50epochs" / "weights" / "last.pt"
    
    resume_flag = False
    if run_last_checkpoint.exists():
        print(f"[*] [RESUME SAFE] Tìm thấy checkpoint dở dang tại: {run_last_checkpoint}. Tiếp tục train...")
        model = RTDETR(str(run_last_checkpoint))
        resume_flag = True
    elif legacy_run_last.exists():
        print(f"[*] [RESUME SAFE] Tìm thấy checkpoint dở dang tại: {legacy_run_last}. Tiếp tục train...")
        model = RTDETR(str(legacy_run_last))
        resume_flag = True
    elif last_checkpoint.exists():
        print(f"[*] [RESUME SAFE] Tìm thấy checkpoint dở dang tại: {last_checkpoint}. Tiếp tục train...")
        model = RTDETR(str(last_checkpoint))
        resume_flag = True
    else:
        print(f"[*] [NEW RUN] Khởi tạo RT-DETR ({model_variant}) mới cho {epochs} epochs...")
        model = RTDETR(model_variant)

    log_file_csv = os.path.join(log_dir, "rtdetr_epoch_logs.csv")
    epoch_logger = EpochLogger(
        log_file_path=log_file_csv,
        model_name="RT-DETR",
        commit_fn=commit_fn,
    )

    def sync_checkpoints():
        """Đồng bộ checkpoint ngay lập tức từ weights/ sang thư mục gốc của checkpoints."""
        for rname in [run_name, "run_50epochs"]:
            run_best = Path(output_dir) / rname / "weights" / "best.pt"
            run_last = Path(output_dir) / rname / "weights" / "last.pt"
            dest_best = Path(output_dir) / "best.pt"
            dest_last = Path(output_dir) / "last.pt"

            if run_best.exists():
                shutil.copy(run_best, dest_best)
            if run_last.exists():
                shutil.copy(run_last, dest_last)
                break

        if commit_fn is not None:
            try:
                commit_fn()
            except Exception as e:
                print(f"[!] Warning commit storage: {e}")

    main_pid = os.getpid()

    # Xử lý Signal khi Modal gửi tín hiệu ngắt / hết thời gian (SIGTERM)
    def handle_termination(signum, frame):
        if os.getpid() != main_pid:
            return
        print(f"\n[!] Nhận tín hiệu dừng (Signal {signum}) do hết thời gian (Timeout). Đang khẩn cấp lưu checkpoint & log...")
        sync_checkpoints()
        epoch_logger.save_csv()
        if commit_fn is not None:
            commit_fn()
        print("[+] Đã lưu toàn bộ tiến độ vào Storage an toàn trước khi dừng container.")
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, handle_termination)
    except Exception:
        pass  # Trong một số môi trường thread không hỗ trợ signal

    epoch_start_time = 0

    def on_train_epoch_start(trainer):
        nonlocal epoch_start_time
        epoch_start_time = time.time()

    def on_fit_epoch_end(trainer):
        """Chạy sau khi epoch kết thúc, validation xong và checkpoint đã được Ultralytics ghi xuống đĩa."""
        elapsed = round(time.time() - epoch_start_time, 2)
        current_epoch = trainer.epoch + 1

        loss_items = getattr(trainer, "loss_items", None)
        box_loss, cls_loss, giou_loss, train_loss = 0.0, 0.0, 0.0, 0.0
        if isinstance(loss_items, dict):
            box_loss = float(loss_items.get("train/box_loss", loss_items.get("box", loss_items.get("box_loss", 0.0))))
            cls_loss = float(loss_items.get("train/cls_loss", loss_items.get("cls", loss_items.get("cls_loss", 0.0))))
            giou_loss = float(loss_items.get("train/dfl_loss", loss_items.get("giou", loss_items.get("giou_loss", loss_items.get("dfl", 0.0)))))
            train_loss = float(loss_items.get("train/loss", loss_items.get("loss", box_loss + cls_loss + giou_loss)))
        elif hasattr(loss_items, "tolist"):
            items_list = loss_items.tolist()
            if isinstance(items_list, list):
                box_loss = float(items_list[0]) if len(items_list) > 0 else 0.0
                cls_loss = float(items_list[1]) if len(items_list) > 1 else 0.0
                giou_loss = float(items_list[2]) if len(items_list) > 2 else 0.0
                train_loss = sum([box_loss, cls_loss, giou_loss])
        elif isinstance(loss_items, (list, tuple)):
            box_loss = float(loss_items[0]) if len(loss_items) > 0 else 0.0
            cls_loss = float(loss_items[1]) if len(loss_items) > 1 else 0.0
            giou_loss = float(loss_items[2]) if len(loss_items) > 2 else 0.0
            train_loss = sum([box_loss, cls_loss, giou_loss])

        if train_loss == 0.0 and hasattr(trainer, "tloss") and trainer.tloss is not None:
            try:
                train_loss = float(trainer.tloss.item() if hasattr(trainer.tloss, "item") else trainer.tloss)
            except Exception:
                pass

        lr = trainer.optimizer.param_groups[0]["lr"] if trainer.optimizer else lr0
        val_metrics = getattr(trainer, "metrics", {}) or {}
        val_p = val_metrics.get("metrics/precision(B)", None)
        val_r = val_metrics.get("metrics/recall(B)", None)
        val_map50 = val_metrics.get("metrics/mAP50(B)", None)
        val_map50_95 = val_metrics.get("metrics/mAP50-95(B)", None)

        # 1. Ghi log
        epoch_logger.log_epoch(
            epoch=current_epoch,
            train_loss=train_loss,
            class_loss=cls_loss,
            box_loss=box_loss,
            giou_loss=giou_loss,
            learning_rate=lr,
            val_precision=val_p,
            val_recall=val_r,
            val_map50=val_map50,
            val_map50_95=val_map50_95,
            elapsed_seconds=elapsed,
        )

        # 2. Đồng bộ best.pt và last.pt ngay sau mỗi epoch
        sync_checkpoints()

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    patience = train_cfg.get("patience", 10)

    print(f"[*] Bắt đầu huấn luyện RT-DETR trên {data_yaml_path} (epochs={epochs}, patience={patience}, resume={resume_flag})...")
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            batch=batch_size,
            lr0=lr0,
            device=device,
            project=output_dir,
            name=run_name,
            exist_ok=True,
            save=True,
            val=True,
            plots=True,
            resume=resume_flag,
        )
    finally:
        sync_checkpoints()
        epoch_logger.save_csv()
        if commit_fn is not None:
            commit_fn()

    dest_best = Path(output_dir) / "best.pt"
    print(f"[+] Hoàn tất huấn luyện RT-DETR! Checkpoint: {dest_best}")
    return str(dest_best)


if __name__ == "__main__":
    train_rtdetr()
