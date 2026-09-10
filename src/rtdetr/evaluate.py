"""RT-DETR Evaluation Module
Đánh giá mô hình RT-DETR trên tập test:
- Tính Precision, Recall, mAP50, mAP50-95, F1
- Trích xuất Ma trận nhầm lẫn (Confusion Matrix)
- Đo đạc phần cứng: Parameters, GFLOPs, Latency (ms), FPS trên GPU A100
- Xuất báo cáo markdown & JSON
"""
import os
import shutil
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from ultralytics import RTDETR
from src.common.metrics_reporter import MetricsReporter
from src.common.benchmark import measure_hardware_benchmark


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_annotated_confusion_matrix(
    matrix: np.ndarray,
    names: List[str],
    save_path: str,
    normalized: bool = True,
    title: str = "",
):
    """Vẽ ma trận nhầm lẫn với số hiển thị rõ ràng trên từng ô (annot=True).
    Chuẩn Machine Learning quốc tế: Trục Tung (Y) = True Label, Trục Hoành (X) = Predicted Label.
    Ultralytics raw_cm lưu theo trục [Predicted, True], do đó cần chuyển vị matrix.T để đúng chuẩn.
    Đồng thời tự động xuất dữ liệu ra file CSV chính xác 100%.
    """
    plt.figure(figsize=(24, 20))
    mat_ml = matrix.T.copy()
    if normalized:
        norm_mat = mat_ml.astype(np.float32)
        row_sums = norm_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = np.round(norm_mat / row_sums, 4)
        fmt = ".2f"
    else:
        mat = mat_ml.astype(int)
        fmt = "d"

    # Xuất ra file CSV chính xác từng chữ số
    csv_path = save_path.replace(".png", ".csv")
    try:
        import pandas as pd
        df_mat = pd.DataFrame(mat, index=names, columns=names)
        df_mat.to_csv(csv_path, index=True)
        print(f"[+] Exported Confusion Matrix CSV: {csv_path}")
    except Exception as e:
        print(f"[!] Warning exporting CSV for {save_path}: {e}")

    sns.heatmap(
        mat,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        cbar=True,
        annot_kws={"size": 6.5},
    )
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def evaluate_rtdetr(
    checkpoint_path: str = "/vol/checkpoints/rtdetr/best.pt",
    data_yaml_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    config_path: str = "configs/rtdetr.yaml",
    output_dir: str = "/vol/outputs/rtdetr_eval",
    **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Tìm checkpoint hợp lệ
    if not os.path.exists(checkpoint_path):
        for alt in [
            "/vol/checkpoints/rtdetr/run_80epochs/weights/best.pt",
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

    # Đọc cấu hình đánh giá nếu có
    eval_conf = 0.001
    cm_conf = 0.25
    imgsz = 640
    for cp in [config_path, "/root/configs/rtdetr.yaml", "configs/rtdetr.yaml"]:
        if os.path.exists(cp):
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                eval_conf = float(cfg.get("evaluation", {}).get("eval_conf", eval_conf))
                cm_conf = float(cfg.get("evaluation", {}).get("cm_conf", cm_conf))
                imgsz = int(cfg.get("training", {}).get("imgsz", imgsz))
                break
            except Exception:
                pass

    print(f"[*] Evaluating RT-DETR using checkpoint: {checkpoint_path} on: {data_yaml_path} (eval_conf={eval_conf}, cm_conf={cm_conf}, imgsz={imgsz})...")
    model = RTDETR(checkpoint_path)

    # 1. Run validation on test split với conf=0.001 (hoặc eval_conf) chuẩn COCO để lấy mAP đầy đủ
    test_run_dir = os.path.join(output_dir, "test_results")
    metrics = model.val(
        data=data_yaml_path,
        split="test",
        conf=eval_conf,
        imgsz=imgsz,
        project=output_dir,
        name="test_results",
        exist_ok=True,
        plots=False,
    )

    # Trích xuất ngưỡng tối ưu F1 (Optimal Confidence Threshold theo chuẩn Everingham et al., IJCV)
    opt_conf = cm_conf
    try:
        from ultralytics.utils.metrics import smooth
        f1_curve = metrics.box.f1_curve
        px = metrics.box.px
        f1_mean = smooth(f1_curve.mean(0), 0.1)
        opt_idx = int(f1_mean.argmax())
        opt_conf = float(px[opt_idx])
        print(f"[+] [OPTIMAL OPERATING POINT] RT-DETR max-F1 threshold: opt_conf = {opt_conf:.3f} (max F1 = {float(f1_mean[opt_idx]):.4f})")
    except Exception as e:
        print(f"[!] Warning extracting optimal F1 conf: {e}, using {cm_conf}")
        opt_conf = cm_conf

    # 2. Run validation tại đúng ngưỡng tối ưu opt_conf để sinh Confusion Matrix đồng bộ 100% với Bảng
    cm_run_dir = os.path.join(output_dir, "cm_results")
    cm_metrics = model.val(
        data=data_yaml_path,
        split="test",
        conf=round(opt_conf, 3),
        imgsz=imgsz,
        project=output_dir,
        name="cm_results",
        exist_ok=True,
        plots=True,
    )
    raw_cm = cm_metrics.confusion_matrix.matrix  # shape (33, 33)

    # Extract overall metrics
    overall_metrics = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    # Extract per-class metrics an toàn
    class_names = metrics.names if hasattr(metrics, "names") else {}
    p_list = metrics.box.p.tolist() if hasattr(metrics.box.p, "tolist") else list(metrics.box.p)
    r_list = metrics.box.r.tolist() if hasattr(metrics.box.r, "tolist") else list(metrics.box.r)
    ap50_list = metrics.box.ap50.tolist() if hasattr(metrics.box.ap50, "tolist") else list(metrics.box.ap50)
    ap_list = metrics.box.ap.tolist() if hasattr(metrics.box.ap, "tolist") else list(metrics.box.ap)

    # Lấy ground truth instances per class từ cột Ground Truth của raw_cm (tương đương hàng của raw_cm.T)
    num_classes = len(class_names)
    nt_list = [0] * num_classes

    if raw_cm is not None and raw_cm.shape[1] >= num_classes:
        gt_col_sums = raw_cm[:, :num_classes].sum(axis=0)
        if gt_col_sums.sum() > 0:
            nt_list = [int(x) for x in gt_col_sums]

    # Cách 2: Lấy từ COCO annotations nếu có
    if not any(nt_list):
        coco_candidates = [
            "/vol/datasets/completed-project-5/coco/test/_annotations.coco.json",
            os.path.join(os.path.dirname(data_yaml_path), "..", "coco", "test", "_annotations.coco.json"),
        ]
        for cpath in coco_candidates:
            if os.path.exists(cpath):
                try:
                    with open(cpath, "r", encoding="utf-8") as f:
                        cdata = json.load(f)
                    cat_map = {c["name"].lower(): c["id"] for c in cdata.get("categories", [])}
                    ann_counts = {}
                    for a in cdata.get("annotations", []):
                        cid = a["category_id"]
                        ann_counts[cid] = ann_counts.get(cid, 0) + 1
                    for idx, cname in enumerate(class_names.values() if isinstance(class_names, dict) else class_names):
                        cid = cat_map.get(str(cname).lower())
                        if cid in ann_counts:
                            nt_list[idx] = ann_counts[cid]
                    break
                except Exception:
                    pass

    per_class_metrics = []
    cnames_list = []
    for i, cname in (class_names.items() if isinstance(class_names, dict) else enumerate(class_names)):
        idx = int(i)
        cname_str = str(cname)
        cnames_list.append(cname_str)
        p_val = float(p_list[idx]) if idx < len(p_list) else 0.0
        r_val = float(r_list[idx]) if idx < len(r_list) else 0.0
        ap50_val = float(ap50_list[idx]) if idx < len(ap50_list) else 0.0
        ap_val = float(ap_list[idx]) if idx < len(ap_list) else 0.0
        inst_val = int(nt_list[idx]) if idx < len(nt_list) else 0

        per_class_metrics.append({
            "Class ID": idx,
            "Class Name": cname_str,
            "Instances": inst_val,
            "Precision": round(0.0 if str(p_val) == "nan" else p_val, 4),
            "Recall": round(0.0 if str(r_val) == "nan" else r_val, 4),
            "mAP@50": round(0.0 if str(ap50_val) == "nan" else ap50_val, 4),
            "mAP@50-95": round(0.0 if str(ap_val) == "nan" else ap_val, 4),
        })

    # 3. Vẽ Confusion Matrix có SỐ hiển thị rõ ràng trên từng ô (annot=True) chuẩn Y=True, X=Pred
    display_names = cnames_list + ["background"]
    cm_counts_path = os.path.join(output_dir, "confusion_matrix.png")
    cm_counts_prefixed = os.path.join(output_dir, "rtdetr_confusion_matrix.png")
    plot_annotated_confusion_matrix(
        matrix=raw_cm,
        names=display_names,
        save_path=cm_counts_path,
        normalized=False,
        title=f"RT-DETR Confusion Matrix (Counts @ opt_conf={opt_conf:.2f})",
    )
    shutil.copy(cm_counts_path, cm_counts_prefixed)

    cm_norm_path = os.path.join(output_dir, "confusion_matrix_normalized.png")
    cm_norm_prefixed = os.path.join(output_dir, "rtdetr_confusion_matrix_normalized.png")
    plot_annotated_confusion_matrix(
        matrix=raw_cm,
        names=display_names,
        save_path=cm_norm_path,
        normalized=True,
        title=f"RT-DETR Normalized Confusion Matrix (@ opt_conf={opt_conf:.2f})",
    )
    cm_counts_csv = cm_counts_path.replace(".png", ".csv")
    cm_counts_prefixed_csv = cm_counts_prefixed.replace(".png", ".csv")
    if os.path.exists(cm_counts_csv):
        shutil.copy(cm_counts_csv, cm_counts_prefixed_csv)

    cm_norm_csv = cm_norm_path.replace(".png", ".csv")
    cm_norm_prefixed_csv = cm_norm_prefixed.replace(".png", ".csv")
    if os.path.exists(cm_norm_csv):
        shutil.copy(cm_norm_csv, cm_norm_prefixed_csv)

    # Xuất file CSV chi tiết từng class (Per-class Metrics CSV)
    per_class_csv = os.path.join(output_dir, "per_class_metrics.csv")
    try:
        import pandas as pd
        pd.DataFrame(per_class_metrics).to_csv(per_class_csv, index=False)
        print(f"[+] Exported RT-DETR per_class_metrics CSV: {per_class_csv}")
    except Exception as e:
        print(f"[!] Warning exporting per_class_metrics CSV: {e}")

    cm_paths = {
        "confusion_matrix.png": cm_counts_path,
        "confusion_matrix_normalized.png": cm_norm_path,
        "confusion_matrix.csv": cm_counts_csv,
        "confusion_matrix_normalized.csv": cm_norm_csv,
        "per_class_metrics.csv": per_class_csv,
    }

    # 4. Đo đạc Benchmark Phần cứng (GFLOPs, Latency ms, FPS)
    hw_bench = measure_hardware_benchmark(
        model_obj=model.model if hasattr(model, "model") else model,
        model_name="RT-DETR",
        imgsz=imgsz,
        warmup_runs=20,
        num_runs=100,
    )

    reporter = MetricsReporter(output_dir=output_dir)
    md_report = reporter.save_test_report(
        model_name="RT-DETR",
        overall_metrics=overall_metrics,
        per_class_metrics=per_class_metrics,
        hardware_benchmark=hw_bench,
        confusion_matrix_paths=cm_paths,
    )
    return overall_metrics, md_report


if __name__ == "__main__":
    evaluate_rtdetr()
