# Huấn Luyện & Đánh Giá RT-DETR và RF-DETR Trên Modal GPU

Dự án nghiên cứu khoa học: Huấn luyện và đánh giá hai kiến trúc mô hình Object Detection dựa trên Transformer (**RT-DETR** và **RF-DETR**) trên tập dữ liệu thực phẩm Roboflow sử dụng hạ tầng điện toán đám mây **Modal GPU**.

---

## 1. Yêu Cầu Kỹ Thuật (Từ `request.md`)

- **Dataset**: `https://app.roboflow.com/nckhcict2025/completed-project/5`
- **Số Epochs**: **50 Epochs**.
- **Loss Tracking từng Epoch**:
  - `Epoch`
  - `Train loss`
  - `Class loss`
  - `Box loss`
  - `GIoU`
  - `Learning rate`
  - `Thời gian chạy thực tế (giây)`
- **Checkpoints**:
  - `best.pt` / `best.pth`: Lưu trọng số tốt nhất theo chỉ số `mAP@50-95` trên tập Validation.
  - `last.pt` / `last.pth`: Lưu trọng số epoch cuối (epoch 50).
- **Đánh Giá & Báo Cáo**:
  - Precision, Recall, mAP50, mAP50-95.
  - Độ chính xác từng lớp (Per-class) trên tập Test.
  - Độ chính xác tổng thể.
  - Bảng so sánh giữa RT-DETR và RF-DETR.

---

## 2. Cấu Trúc Thư Mục

```text
Train_models/
├── configs/
│   ├── dataset.yaml            # Cấu hình Roboflow URL & paths
│   ├── rtdetr.yaml             # Cấu hình train RT-DETR (50 epochs)
│   └── rfdetr.yaml             # Cấu hình train RF-DETR (50 epochs)
├── src/
│   ├── data/
│   │   ├── roboflow_download.py # Tải dataset YOLO và COCO
│   │   └── validate_dataset.py  # Kiểm tra tính toàn vẹn và phân bố nhãn
│   ├── common/
│   │   ├── epoch_logger.py      # Logger bóc tách 6 thông số từng epoch
│   │   └── metrics_reporter.py  # Đánh giá Precision, Recall, mAP, per-class report
│   ├── rtdetr/
│   │   ├── train.py             # Script train RT-DETR
│   │   └── evaluate.py          # Script đánh giá RT-DETR trên Test set
│   └── rfdetr/
│       ├── train.py             # Script train RF-DETR
│       └── evaluate.py          # Script đánh giá RF-DETR trên Test set
├── modal_app/
│   └── pipeline.py             # Điều phối chạy toàn bộ pipeline
├── requirements.txt            # Danh sách thư viện
├── request.md                  # Yêu cầu gốc
└── TRAINING_PLAN.md            # Kế hoạch chi tiết & nhật ký tiến độ
```

---

## 3. Hướng Dẫn Chạy Trên Modal GPU

Vì Modal Secret `roboflow-secret` đã được tạo sẵn trên tài khoản của bạn, các lệnh chạy cực kỳ đơn giản và không cần truyền lại API key.

### Chạy Toàn Bộ Pipeline Tự Động (Download + Train RT-DETR + Train RF-DETR + Eval):
```bash
modal run modal_app/pipeline.py
```

### Hoặc Chạy Từng Bước Độc Lập:
```bash
# 1. Tải và kiểm tra dataset (v5)
modal run modal_app/pipeline.py::download_dataset_step

# 2. Train RT-DETR (50 epochs)
modal run modal_app/pipeline.py::train_rtdetr_step

# 3. Train RF-DETR (50 epochs)
modal run modal_app/pipeline.py::train_rfdetr_step

# 4. Đánh giá và xuất bảng so sánh
modal run modal_app/pipeline.py::evaluate_both_models
```

---

## 4. Vị Trí Lưu Trữ Dữ Liệu Trên Modal Volume (`food-detection-training`)

- **Dataset**: `/vol/datasets/completed-project-5/`
- **Checkpoints RT-DETR**: `/vol/checkpoints/rtdetr/best.pt`, `/vol/checkpoints/rtdetr/last.pt`
- **Checkpoints RF-DETR**: `/vol/checkpoints/rfdetr/best.pth`, `/vol/checkpoints/rfdetr/last.pth`
- **Logs từng Epoch**: `/vol/logs/rtdetr_epoch_logs.csv`, `/vol/logs/rfdetr_epoch_logs.csv`
- **Báo cáo so sánh**: `/vol/logs/comparison_report.md`
