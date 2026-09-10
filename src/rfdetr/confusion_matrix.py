"""RF-DETR Confusion Matrix & Authentic COCO Evaluation Module
Tạo và trực quan hóa Ma trận nhầm lẫn (Confusion Matrix) & Đánh giá COCO chuẩn cho RF-DETR:
- Tính toán ma trận (C + 1) x (C + 1) bao gồm 32 classes thực phẩm + Background (FP / FN)
- Greedy IoU Matching (ngưỡng IoU = 0.5) giữa detection và ground-truth tại ngưỡng conf = 0.25
- Xuất 2 biểu đồ nhiệt heatmap với số hiển thị rõ ràng trên từng ô (annot=True):
  1. Counts Matrix (số lượng cá thể dự đoán)
  2. Normalized Matrix (chuẩn hóa tỷ lệ % theo từng lớp thực tế)
- Tích hợp pycocotools COCOeval để tính mAP@50, mAP@50-95, Precision, Recall và Instances thực tế 100% cho từng class
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm


def box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """Tính IoU giữa 2 tập bounding box [N, 4] và [M, 4] dạng [x1, y1, x2, y2]."""
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)))

    lt = np.maximum(box1[:, None, :2], box2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(box1[:, None, 2:], box2[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, a_min=0, a_max=None)              # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]                       # [N, M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / np.clip(union, a_min=1e-7, a_max=None)


def compute_rfdetr_confusion_matrix(
    model: Any,
    dataset_dir: str,
    output_dir: str,
    class_names: List[str],
    conf_threshold: float = 0.25,
    eval_conf: float = 0.01,
    iou_threshold: float = 0.5,
    max_test_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Sinh ma trận nhầm lẫn chi tiết và tính toán chỉ số COCO thực tế cho RF-DETR trên tập Test."""
    os.makedirs(output_dir, exist_ok=True)
    num_classes = len(class_names)
    bg_idx = num_classes  # Lớp background nằm ở chỉ số cuối cùng
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

    # 1. Tìm file COCO test annotation
    test_json_candidates = [
        os.path.join(dataset_dir, "test", "_annotations.coco.json"),
        os.path.join(dataset_dir, "valid", "_annotations.coco.json"),
        os.path.join(dataset_dir, "val", "_annotations.coco.json"),
    ]
    test_json_path = None
    for cand in test_json_candidates:
        if os.path.exists(cand):
            test_json_path = cand
            break

    if not test_json_path:
        print(f"[!] Không tìm thấy file annotations.coco.json trong {dataset_dir} để đánh giá.")
        return {}

    print(f"[*] Đang nạp annotations COCO từ: {test_json_path}...")
    with open(test_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Ánh xạ category_id -> class_idx (0..num_classes-1)
    cats = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
    cat_to_idx = {}
    for cat_id, cname in cats.items():
        if cname in class_names:
            cat_to_idx[cat_id] = class_names.index(cname)
        else:
            for idx, n in enumerate(class_names):
                if n.lower() == cname.lower():
                    cat_to_idx[cat_id] = idx
                    break

    idx_to_cat = {idx: cat_id for cat_id, idx in cat_to_idx.items()}

    # Nhóm ground truths theo image_id
    img_dir = os.path.dirname(test_json_path)
    annotations_by_img = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in annotations_by_img:
            annotations_by_img[img_id] = []
        annotations_by_img[img_id].append(ann)

    images = coco_data.get("images", [])
    if max_test_samples and len(images) > max_test_samples:
        images = images[:max_test_samples]

    print(f"[*] Bắt đầu suy luận & đánh giá RF-DETR trên toàn bộ {len(images)} ảnh test (eval_conf={eval_conf}, cm_conf={conf_threshold})...")

    coco_predictions = []

    # 2. Duyệt qua từng ảnh và thực hiện inference
    for i, img_info in enumerate(tqdm(images, desc="[RF-DETR Evaluation]")):
        img_file = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(img_file):
            alt_img = os.path.join(img_dir, os.path.basename(img_info["file_name"]))
            if os.path.exists(alt_img):
                img_file = alt_img
            else:
                continue

        # Lấy ground truth boxes [N, 4] dạng [x1, y1, x2, y2]
        gt_boxes = []
        gt_classes = []
        for ann in annotations_by_img.get(img_info["id"], []):
            cid = ann.get("category_id")
            if cid in cat_to_idx:
                x, y, w, h = ann["bbox"]
                gt_boxes.append([x, y, x + w, y + h])
                gt_classes.append(cat_to_idx[cid])

        gt_boxes = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)
        gt_classes = np.array(gt_classes, dtype=np.int32) if gt_classes else np.zeros((0,), dtype=np.int32)

        # Dự đoán từ RF-DETR model với ngưỡng thấp eval_conf (0.01) để lấy toàn bộ dự đoán cho COCOeval
        all_pred_boxes = []
        all_pred_classes = []
        all_pred_scores = []
        try:
            detections = model.predict(img_file, conf_threshold=eval_conf)
            if hasattr(detections, "xyxy"):
                for box, cid, score in zip(detections.xyxy, detections.class_id, detections.confidence):
                    cid_int = int(cid)
                    if cid_int < num_classes:
                        all_pred_boxes.append(box)
                        all_pred_classes.append(cid_int)
                        all_pred_scores.append(float(score))
            elif isinstance(detections, dict):
                boxes = detections.get("boxes", [])
                classes = detections.get("labels", detections.get("classes", []))
                scores = detections.get("scores", [])
                for b, c, s in zip(boxes, classes, scores):
                    c_int = int(c)
                    if c_int < num_classes:
                        all_pred_boxes.append(b)
                        all_pred_classes.append(c_int)
                        all_pred_scores.append(float(s))
        except Exception:
            pass

        # Thu thập detections cho COCOeval
        for b, c, s in zip(all_pred_boxes, all_pred_classes, all_pred_scores):
            if c in idx_to_cat:
                coco_cat_id = idx_to_cat[c]
                x1, y1, x2, y2 = b
                w = max(0.0, float(x2 - x1))
                h = max(0.0, float(y2 - y1))
                coco_predictions.append({
                    "image_id": int(img_info["id"]),
                    "category_id": int(coco_cat_id),
                    "bbox": [round(float(x1), 2), round(float(y1), 2), round(w, 2), round(h, 2)],
                    "score": round(float(s), 4),
                })

        # Lọc detections cho Confusion Matrix tại ngưỡng conf_threshold (0.25)
        cm_pred_boxes = []
        cm_pred_classes = []
        for b, c, s in zip(all_pred_boxes, all_pred_classes, all_pred_scores):
            if s >= conf_threshold:
                cm_pred_boxes.append(b)
                cm_pred_classes.append(c)

        pred_boxes = np.array(cm_pred_boxes, dtype=np.float32) if cm_pred_boxes else np.zeros((0, 4), dtype=np.float32)
        pred_classes = np.array(cm_pred_classes, dtype=np.int32) if cm_pred_classes else np.zeros((0,), dtype=np.int32)

        # 3. Matching cho Confusion Matrix
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue

        if len(gt_boxes) == 0 and len(pred_boxes) > 0:
            for pc in pred_classes:
                matrix[bg_idx, pc] += 1
            continue

        if len(pred_boxes) == 0 and len(gt_boxes) > 0:
            for gc in gt_classes:
                matrix[gc, bg_idx] += 1
            continue

        ious = box_iou(gt_boxes, pred_boxes)
        matched_gt = set()
        matched_pred = set()

        # Greedy matching theo thứ tự IoU cao nhất
        pairs = []
        for g_idx in range(len(gt_boxes)):
            for p_idx in range(len(pred_boxes)):
                if ious[g_idx, p_idx] >= iou_threshold:
                    pairs.append((ious[g_idx, p_idx], g_idx, p_idx))

        pairs.sort(key=lambda x: x[0], reverse=True)
        for iou_val, g_idx, p_idx in pairs:
            if g_idx not in matched_gt and p_idx not in matched_pred:
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)
                gc = gt_classes[g_idx]
                pc = pred_classes[p_idx]
                matrix[gc, pc] += 1

        # GT không được dự đoán -> False Negative
        for g_idx in range(len(gt_boxes)):
            if g_idx not in matched_gt:
                matrix[gt_classes[g_idx], bg_idx] += 1

        # Pred không có GT khớp -> False Positive
        for p_idx in range(len(pred_boxes)):
            if p_idx not in matched_pred:
                matrix[bg_idx, pred_classes[p_idx]] += 1

    # 4. Trực quan hóa Ma trận nhầm lẫn với số hiển thị rõ ràng trên từng ô (annot=True)
    display_names = class_names + ["background"]
    
    # A. Ma trận Counts
    plt.figure(figsize=(24, 20))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        cbar=True,
        annot_kws={"size": 6.5},
    )
    plt.title(f"RF-DETR Confusion Matrix (Counts @ conf={conf_threshold})", fontsize=16, pad=15)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    counts_png = os.path.join(output_dir, "confusion_matrix.png")
    rfdetr_counts_png = os.path.join(output_dir, "rfdetr_confusion_matrix.png")
    plt.savefig(counts_png, dpi=200)
    plt.savefig(rfdetr_counts_png, dpi=200)
    plt.close()

    # B. Ma trận Normalized (% theo True Label)
    norm_matrix = matrix.astype(np.float32)
    row_sums = norm_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm_matrix = np.round(norm_matrix / row_sums, 2)

    plt.figure(figsize=(24, 20))
    sns.heatmap(
        norm_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        cbar=True,
        annot_kws={"size": 6.5},
    )
    plt.title(f"RF-DETR Normalized Confusion Matrix (@ conf={conf_threshold})", fontsize=16, pad=15)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    norm_png = os.path.join(output_dir, "confusion_matrix_normalized.png")
    rfdetr_norm_png = os.path.join(output_dir, "rfdetr_confusion_matrix_normalized.png")
    plt.savefig(norm_png, dpi=200)
    plt.savefig(rfdetr_norm_png, dpi=200)
    plt.close()

    print(f"[+] RF-DETR Confusion Matrix đã xuất thành công với số hiển thị rõ ràng:\n - {counts_png}\n - {norm_png}")

    # 5. Chạy COCOeval để tính toán chỉ số mAP per-class thực tế
    per_class_data = []
    overall_data = {
        "precision": 0.0,
        "recall": 0.0,
        "map50": 0.0,
        "map50_95": 0.0,
    }

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(test_json_path)
        if len(coco_predictions) > 0:
            coco_dt = coco_gt.loadRes(coco_predictions)
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
            target_cat_ids = [idx_to_cat[i] for i in range(num_classes) if i in idx_to_cat]
            coco_eval.params.catIds = target_cat_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            overall_data["map50_95"] = float(coco_eval.stats[0])
            overall_data["map50"] = float(coco_eval.stats[1])

            # Tính micro-average Precision & Recall từ Confusion Matrix
            total_tp = int(np.trace(matrix[:num_classes, :num_classes]))
            total_fp = int(matrix[bg_idx, :num_classes].sum() + (matrix[:num_classes, :num_classes].sum() - total_tp))
            total_fn = int(matrix[:num_classes, bg_idx].sum() + (matrix[:num_classes, :num_classes].sum() - total_tp))
            overall_data["precision"] = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
            overall_data["recall"] = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0

            for k in range(len(target_cat_ids)):
                cat_id = target_cat_ids[k]
                cname = class_names[k]
                inst = len(coco_gt.getAnnIds(catIds=[cat_id]))

                # mAP@50-95
                prec_all = coco_eval.eval["precision"][:, :, k, 0, 2]
                valid_p = prec_all[prec_all > -1]
                cls_map50_95 = float(np.mean(valid_p)) if len(valid_p) > 0 else 0.0

                # mAP@50
                prec50 = coco_eval.eval["precision"][0, :, k, 0, 2]
                valid_p50 = prec50[prec50 > -1]
                cls_map50 = float(np.mean(valid_p50)) if len(valid_p50) > 0 else 0.0

                # Precision & Recall từ Confusion Matrix tại conf=0.25
                tp = int(matrix[k, k])
                fp = int(matrix[:, k].sum() - tp)
                fn = int(matrix[k, :].sum() - tp)
                cls_p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                cls_r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

                per_class_data.append({
                    "Class ID": k,
                    "Class Name": cname,
                    "Instances": inst,
                    "Precision": round(cls_p, 4),
                    "Recall": round(cls_r, 4),
                    "mAP@50": round(cls_map50, 4),
                    "mAP@50-95": round(cls_map50_95, 4),
                })
    except Exception as e:
        print(f"[!] Warning running COCOeval on RF-DETR predictions: {e}")
        import traceback
        traceback.print_exc()

    return {
        "confusion_matrix": counts_png,
        "confusion_matrix_normalized": norm_png,
        "matrix": matrix,
        "norm_matrix": norm_matrix,
        "per_class": per_class_data,
        "overall": overall_data,
        "coco_predictions": coco_predictions,
    }
