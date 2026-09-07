# Huấn Luyện & Đánh Giá RT-DETR và RF-DETR Trên Modal GPU

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Modal](https://img.shields.io/badge/Cloud-Modal%20GPU%20(A100)-green.svg)](https://modal.com/)
[![Roboflow](https://img.shields.io/badge/Dataset-Roboflow%20v5-purple.svg)](https://app.roboflow.com/nckhcict2025/completed-project/5)
[![Checkpoints](https://img.shields.io/badge/Google%20Drive-Model%20Weights%20v5-4285F4.svg?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/18o5EigDe8EyL2dcgpFc1YnOiUw9qhs9f?usp=drive_link)

Dự án nghiên cứu khoa học: Huấn luyện, tối ưu và đánh giá đối đầu hai kiến trúc mô hình Object Detection dựa trên Transformer tiên tiến (**RT-DETR** và **RF-DETR**) trên tập dữ liệu 32 lớp món ăn & nguyên liệu thực phẩm Việt Nam sử dụng hạ tầng điện toán đám mây **Modal GPU (NVIDIA A100)**.

---

## 1. Tải Trọng Số Đã Huấn Luyện (Model Checkpoints v5)

Toàn bộ tệp trọng số (checkpoints) tốt nhất và đầy đủ của cả hai mô hình trên phiên bản dữ liệu **v5** đã được lưu trữ trên Google Drive:

🔗 **[Tải Trọn Bộ Checkpoints v5 Trên Google Drive](https://drive.google.com/drive/folders/18o5EigDe8EyL2dcgpFc1YnOiUw9qhs9f?usp=drive_link)**

### Danh Sách Chi Tiết Tệp Trọng Số:

| Mô hình | Tệp Checkpoint | Dung lượng | Mô tả & Chỉ số đạt được (Test Split v5) |
| :--- | :--- | :---: | :--- |
| **RT-DETR** | `rtdetr/best.pt` | **63.3 MB** | 🏆 **Trọng số tốt nhất** (mAP@50: **98.56%**, mAP@50-95: **88.45%**) |
| **RT-DETR** | `rtdetr/last.pt` | 63.3 MB | Trọng số epoch 50 cuối cùng |
| **RF-DETR** | `rfdetr/best.pth` | **128.0 MB** | 🏆 **Trọng số EMA tốt nhất** (mAP@50: **98.64%**, mAP@50-95: **87.93%**) |
| **RF-DETR** | `rfdetr/last.pth` | 511.8 MB | Trọng số checkpoint đầy đủ epoch 35 |
| **RF-DETR** | `rfdetr/last_ema.pth` | 128.0 MB | Trọng số EMA epoch 35 |
| **RF-DETR** | `rfdetr/checkpoint_*.ckpt` | ~511 MB / file | Checkpoint trung gian (epoch 9, 19, 29) phục vụ resume training |

> **Hướng dẫn sử dụng sau khi tải**:
> Sau khi tải thư mục từ Google Drive về, bạn đặt vào đường dẫn dự án theo cấu trúc:
> ```text
> synced_results/checkpoints/checkpoints/
> ├── rtdetr/
> │   ├── best.pt
> │   └── last.pt
> └── rfdetr/
>     ├── best.pth
>     └── last.pth
> ```

---

## 2. Bảng Tổng Hợp Kết Quả Đối Đầu (Benchmark v5)

Đo lường trên tập kiểm thử độc lập **Test Split v5** (1,481 ảnh, 3,911 instances trên 32 lớp thực phẩm):

| Tiêu Chí So Sánh (Evaluation Metric) | RT-DETR (Ultralytics) | RF-DETR (Medium) | Chênh Lệch | Ưu Thế |
| :--- | :---: | :---: | :---: | :---: |
| **Precision (Độ chính xác)** | 97.56% | **97.60%** | +0.04% | 🏆 RF-DETR |
| **Recall (Độ nhạy / Thu hồi)** | 97.59% | **97.82%** | +0.23% | 🏆 RF-DETR |
| **F1-Score** | 97.57% | **97.69%** | +0.12% | 🏆 RF-DETR |
| **mAP@50 (IoU = 0.50)** | 98.56% | **98.64%** | +0.08% | 🏆 RF-DETR |
| **mAP@50-95 (COCO Standard)** | **88.45%** | 87.93% | +0.52% | 🏆 RT-DETR |
| **mAP@75 (IoU = 0.75)** | 95.12% | **95.42%** | +0.30% | 🏆 RF-DETR |
| **Tốc độ suy luận (Inference Latency)** | **5.4 ms / ảnh (~185 FPS)** | ~12.5 ms / ảnh (~80 FPS) | Nhanh hơn 2.3x | 🏆 RT-DETR |
| **Kích thước mô hình (Model Size)** | **63.4 MB** | 134.2 MB | Nhẹ hơn 2.1x | 🏆 RT-DETR |

* Báo cáo so sánh đối đầu chi tiết: [reports_v5/COMPARISON_REPORT.md](reports_v5/COMPARISON_REPORT.md)
* Báo cáo huấn luyện RT-DETR: [reports_v5/RTDETR_TRAINING_REPORT.md](reports_v5/RTDETR_TRAINING_REPORT.md)
* Báo cáo huấn luyện RF-DETR: [reports_v5/RFDETR_TRAINING_REPORT.md](reports_v5/RFDETR_TRAINING_REPORT.md)

---

## 3. Cấu Trúc Dự Án

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
│   ├── app.py                  # Cấu hình Modal App & Image dependencies
│   └── pipeline.py             # Điều phối chạy toàn bộ pipeline
├── reports_v5/                 # Báo cáo huấn luyện & đối đầu chi tiết v5
│   ├── COMPARISON_REPORT.md
│   ├── RTDETR_TRAINING_REPORT.md
│   └── RFDETR_TRAINING_REPORT.md
├── scripts/
│   └── fix_test_reports.py     # Tiện ích chuẩn hóa báo cáo đánh giá
├── requirements.txt            # Danh sách thư viện cần thiết
├── sync_storage.py             # Script đồng bộ dữ liệu tự động từ Modal Volume
├── RUN_GUIDE.md                # Hướng dẫn chi tiết từng câu lệnh vận hành
└── TRAINING_PLAN.md            # Kế hoạch chi tiết & nhật ký tiến độ
```

---

## 4. Hướng Dẫn Vận Hành Trên Modal GPU

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

## 5. Đồng Bộ Dữ Liệu Từ Modal Volume Về Máy Tính

Dự án tích hợp sẵn công cụ đồng bộ dữ liệu nhanh từ Modal Cloud Volume `food-detection-training`:

```bash
python sync_storage.py
```
Hệ thống sẽ tự động kéo logs, metrics CSV, báo cáo markdown và checkpoints về thư mục cục bộ `synced_results/`.
