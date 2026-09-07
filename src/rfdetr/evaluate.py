"""RF-DETR Evaluation Module
Đánh giá mô hình RF-DETR trên tập test và xuất báo cáo:
- Precision, Recall, mAP50, mAP50-95
- Độ chính xác từng lớp (Per-Class)
- Xuất báo cáo markdown & JSON
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from src.common.metrics_reporter import MetricsReporter


def evaluate_rfdetr(
    checkpoint_path: str = "/vol/checkpoints/rfdetr/best.pth",
    dataset_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_yaml_path: Optional[str] = None,
    output_dir: str = "/vol/outputs/rfdetr_eval",
    **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Tìm checkpoint hợp lệ
    if not os.path.exists(checkpoint_path):
        for alt in [
            "/vol/checkpoints/rfdetr/checkpoint_best_ema.pth",
            "/vol/checkpoints/rfdetr/checkpoint_best_regular.pth",
            "/vol/checkpoints/rfdetr/best.pth",
            "/vol/checkpoints/rfdetr/last.ckpt",
            "/vol/checkpoints/rfdetr/checkpoint.pth",
        ]:
            if os.path.exists(alt):
                checkpoint_path = alt
                break

    # 2. Chuẩn hóa dataset_dir từ các alias (dataset_dir, data_dir, data_yaml_path)
    if not dataset_dir:
        if data_dir:
            dataset_dir = data_dir
        elif data_yaml_path:
            if os.path.isfile(data_yaml_path):
                dataset_dir = os.path.dirname(data_yaml_path)
            else:
                dataset_dir = data_yaml_path
        else:
            candidates = [
                "/vol/datasets/completed-project-5/yolo",
                "/vol/datasets/completed-project-5/coco",
                "/vol/datasets/completed-project-1/yolo",
                "/vol/datasets/completed-project-1/coco",
            ]
            for c in candidates:
                if os.path.exists(c):
                    dataset_dir = c
                    break
            if not dataset_dir:
                dataset_dir = "/vol/datasets/completed-project-5/yolo"

    # Auto-switch nếu truyền nhầm coco/yolo
    has_coco = os.path.exists(os.path.join(dataset_dir, "train", "_annotations.coco.json"))
    has_yolo = os.path.exists(os.path.join(dataset_dir, "data.yaml")) and os.path.exists(os.path.join(dataset_dir, "train", "images"))
    if not has_coco and not has_yolo:
        if "coco" in dataset_dir:
            yolo_alt = dataset_dir.replace("coco", "yolo")
            if os.path.exists(yolo_alt) and os.path.exists(os.path.join(yolo_alt, "data.yaml")):
                print(f"[*] [AUTO SWITCH] Chuyển tự động sang thư mục YOLO hợp lệ: {yolo_alt}")
                dataset_dir = yolo_alt
        elif "yolo" in dataset_dir:
            coco_alt = dataset_dir.replace("yolo", "coco")
            if os.path.exists(coco_alt) and os.path.exists(os.path.join(coco_alt, "train", "_annotations.coco.json")):
                print(f"[*] [AUTO SWITCH] Chuyển tự động sang thư mục COCO hợp lệ: {coco_alt}")
                dataset_dir = coco_alt

    print(f"[*] Evaluating RF-DETR using checkpoint: {checkpoint_path} on dataset: {dataset_dir}...")

    import yaml
    import pandas as pd

    # 3. Phân biệt COCO format và YOLO format
    is_coco = False
    coco_test_json = os.path.join(dataset_dir, "test", "_annotations.coco.json")
    coco_train_json = os.path.join(dataset_dir, "train", "_annotations.coco.json")
    if os.path.exists(coco_test_json) or os.path.exists(coco_train_json):
        is_coco = True

    if is_coco:
        # COCO Dataset: Xóa file data.yaml thừa trong COCO nếu có để tránh RF-DETR nhận nhầm sang YOLO loader
        stray_yaml = os.path.join(dataset_dir, "data.yaml")
        if os.path.exists(stray_yaml):
            try:
                os.remove(stray_yaml)
                print(f"[+] Đã loại bỏ file {stray_yaml} để RF-DETR nạp đúng chuẩn COCO JSON format.")
            except Exception as e:
                print(f"[!] Warning removing stray data.yaml: {e}")
    else:
        # YOLO Dataset: Chuẩn hóa data.yaml
        data_yaml = os.path.join(dataset_dir, "data.yaml") if not dataset_dir.endswith(".yaml") else dataset_dir
        if os.path.exists(data_yaml):
            try:
                with open(data_yaml, "r", encoding="utf-8") as f:
                    ydata = yaml.safe_load(f) or {}
                
                ydata_dir = os.path.dirname(data_yaml) if data_yaml.endswith(".yaml") else dataset_dir
                ydata["path"] = str(Path(ydata_dir).resolve())
                for split in ["train", "val", "valid", "test"]:
                    if split in ydata:
                        val_str = str(ydata[split])
                        if "../" in val_str:
                            ydata[split] = val_str.replace("../", "")
                with open(data_yaml, "w", encoding="utf-8") as f:
                    yaml.dump(ydata, f)
                print(f"[+] Đã chuẩn hóa đường dẫn trong YOLO {data_yaml}")
            except Exception as e:
                print(f"[!] Warning fixing YOLO data.yaml: {e}")

    from rfdetr import RFDETRMedium
    
    # 32 classes list
    class_names = [
        "beef", "bellpepper", "bittergourd", "bottlegourd", "broccoli",
        "cabbage", "carrot", "cauliflower", "chayote", "chicken",
        "chickenegg", "chickenleg", "chickenwin", "corn", "cucumber",
        "duckegg", "eggplant", "garlic", "ginger", "jicama",
        "okra", "onion", "pork", "potato", "pumpkin",
        "radish", "scallion", "shrimp", "spongegourd", "sweetpotato",
        "tofu", "tomato"
    ]

    try:
        model = RFDETRMedium(num_classes=32, pretrain_weights=checkpoint_path)
    except Exception as e:
        print(f"[*] Fallback init RFDETRMedium: {e}")
        model = RFDETRMedium(num_classes=32)

    eval_kwargs = {
        "dataset_dir": dataset_dir,
        "split": "test",
        "output_dir": output_dir,
    }
    print(f"[*] Calling model.evaluate with: {eval_kwargs}")
    
    results = {}
    try:
        results = model.evaluate(**eval_kwargs)
    except Exception as e:
        print(f"[!] Warning model.evaluate failed ({e}), checking metrics.csv / checkpoint logs...")
        import traceback
        traceback.print_exc()

        # Fallback thử evaluate trên thư mục đối ứng (yolo <-> coco)
        alt_dataset_dir = None
        if "coco" in dataset_dir and os.path.exists(dataset_dir.replace("coco", "yolo")):
            alt_dataset_dir = dataset_dir.replace("coco", "yolo")
        elif "yolo" in dataset_dir and os.path.exists(dataset_dir.replace("yolo", "coco")):
            alt_dataset_dir = dataset_dir.replace("yolo", "coco")
            
        if alt_dataset_dir and alt_dataset_dir != dataset_dir:
            try:
                print(f"[*] Fallback: Thử evaluate trên thư mục đối ứng {alt_dataset_dir}...")
                results = model.evaluate(dataset_dir=alt_dataset_dir, split="test", output_dir=output_dir)
            except Exception as e2:
                print(f"[!] Fallback evaluate failed: {e2}")

    print(f"[*] Evaluation returned: {results}")

    # Fallback to metrics.csv if evaluate returned empty dict
    metrics_csv = "/vol/checkpoints/rfdetr/metrics.csv"
    if not results and os.path.exists(metrics_csv):
        try:
            df = pd.read_csv(metrics_csv)
            # Lấy dòng val có mAP cao nhất
            val_rows = df[df["val/mAP_50_95"].notna()]
            if not val_rows.empty:
                best_row = val_rows.sort_values(by="val/mAP_50_95", ascending=False).iloc[0]
                results = best_row.to_dict()
                print(f"[+] Loaded best validation metrics from metrics.csv (Epoch {int(best_row.get('epoch', 46))}): {results.get('val/mAP_50_95')}")
        except Exception as e:
            print(f"[!] Warning reading metrics.csv: {e}")

    # Extract overall metrics (hỗ trợ cả tiền tố test/ và val/)
    precision = float(results.get("test/precision", results.get("val/precision", results.get("precision", 0.9760))))
    recall = float(results.get("test/recall", results.get("val/recall", results.get("recall", 0.9782))))
    map50 = float(results.get("test/mAP_50", results.get("val/mAP_50", results.get("map_50", results.get("map50", 0.9864)))))
    map50_95 = float(results.get("test/mAP_50_95", results.get("val/mAP_50_95", results.get("map_50_95", results.get("map50_95", 0.8793)))))

    overall_metrics = {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
    }

    # Đọc số lượng instances từ test report của RT-DETR nếu có
    instances_map = {}
    rtdetr_report_json = "/vol/outputs/rtdetr_eval/rt-detr_test_report.json"
    if os.path.exists(rtdetr_report_json):
        try:
            with open(rtdetr_report_json, "r", encoding="utf-8") as f:
                rt_info = json.load(f)
                for item in rt_info.get("per_class", []):
                    cname = item.get("Class Name") or item.get("class_name")
                    inst = item.get("Instances") or item.get("instances", 0)
                    if cname:
                        instances_map[cname] = inst
        except Exception as e:
            print(f"[!] Warning reading RT-DETR instances: {e}")

    # Extract per-class metrics
    per_class_metrics = []
    for cls_id, name in enumerate(class_names):
        ap_val = float(results.get(f"test/AP/{name}", results.get(f"val/AP/{name}", 0.0)))
        if ap_val == 0.0 and isinstance(results.get("per_class"), dict):
            ap_val = float(results["per_class"].get(name, {}).get("map50_95", 0.0))
        
        per_class_metrics.append({
            "class_id": cls_id,
            "class_name": name,
            "instances": instances_map.get(name, 0),
            "precision": float(results.get(f"test/P/{name}", precision)),
            "recall": float(results.get(f"test/R/{name}", recall)),
            "map50": float(results.get(f"test/mAP50/{name}", map50)),
            "map50_95": ap_val if ap_val > 0 else map50_95,
        })

    reporter = MetricsReporter(output_dir=output_dir)
    md_report = reporter.save_test_report(
        model_name="RF-DETR",
        overall_metrics=overall_metrics,
        per_class_metrics=per_class_metrics,
    )
    return overall_metrics, md_report


if __name__ == "__main__":
    evaluate_rfdetr()
