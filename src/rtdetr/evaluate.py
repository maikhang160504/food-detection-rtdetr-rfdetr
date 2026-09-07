"""RT-DETR Evaluation Module
Đánh giá mô hình RT-DETR trên tập test:
- Tính Precision, Recall, mAP50, mAP50-95
- Độ chính xác từng lớp (Per-Class)
- Xuất báo cáo markdown & JSON
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from ultralytics import RTDETR
from src.common.metrics_reporter import MetricsReporter


def evaluate_rtdetr(
    checkpoint_path: str = "/vol/checkpoints/rtdetr/best.pt",
    data_yaml_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    output_dir: str = "/vol/outputs/rtdetr_eval",
    **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Tìm checkpoint hợp lệ
    if not os.path.exists(checkpoint_path):
        for alt in [
            "/vol/checkpoints/rtdetr/run_50epochs/weights/best.pt",
            "/vol/checkpoints/rtdetr/best.pt",
            "/vol/checkpoints/rtdetr/last.pt",
        ]:
            if os.path.exists(alt):
                checkpoint_path = alt
                break

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 2. Chuẩn hóa data_yaml_path từ các alias
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

    print(f"[*] Evaluating RT-DETR using checkpoint: {checkpoint_path} on: {data_yaml_path}...")
    model = RTDETR(checkpoint_path)

    # Run validation on test split
    metrics = model.val(data=data_yaml_path, split="test", project=output_dir, name="test_results", exist_ok=True)

    # Extract overall metrics
    overall_metrics = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    # Extract per-class metrics an toàn
    per_class_metrics = []
    class_names = metrics.names if hasattr(metrics, "names") else {}

    p_list = metrics.box.p.tolist() if hasattr(metrics.box.p, "tolist") else list(metrics.box.p)
    r_list = metrics.box.r.tolist() if hasattr(metrics.box.r, "tolist") else list(metrics.box.r)
    ap50_list = metrics.box.ap50.tolist() if hasattr(metrics.box.ap50, "tolist") else list(metrics.box.ap50)
    ap_list = metrics.box.ap.tolist() if hasattr(metrics.box.ap, "tolist") else list(metrics.box.ap)

    for i, cname in (class_names.items() if isinstance(class_names, dict) else enumerate(class_names)):
        idx = int(i)
        p_val = float(p_list[idx]) if idx < len(p_list) else 0.0
        r_val = float(r_list[idx]) if idx < len(r_list) else 0.0
        ap50_val = float(ap50_list[idx]) if idx < len(ap50_list) else 0.0
        ap_val = float(ap_list[idx]) if idx < len(ap_list) else 0.0

        per_class_metrics.append({
            "Class ID": idx,
            "Class Name": str(cname),
            "Precision": round(0.0 if str(p_val) == "nan" else p_val, 4),
            "Recall": round(0.0 if str(r_val) == "nan" else r_val, 4),
            "mAP@50": round(0.0 if str(ap50_val) == "nan" else ap50_val, 4),
            "mAP@50-95": round(0.0 if str(ap_val) == "nan" else ap_val, 4),
        })

    reporter = MetricsReporter(output_dir=output_dir)
    md_report = reporter.save_test_report(
        model_name="RT-DETR",
        overall_metrics=overall_metrics,
        per_class_metrics=per_class_metrics,
    )
    return overall_metrics, md_report


if __name__ == "__main__":
    evaluate_rtdetr()
