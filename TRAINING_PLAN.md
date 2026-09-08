# TRAINING_PLAN.md

> Tài liệu trạng thái dự án dành cho AI và người phát triển.
> Mục đích: lưu yêu cầu, kế hoạch, cấu trúc dự án, tiến độ, lỗi và quyết định kỹ thuật để có thể đọc lại và tiếp tục công việc đồng bộ, minh bạch và nhất quán.
>
> Trạng thái hiện tại: **ĐÃ ĐỒNG BỘ YÊU CẦU & SẴN SÀNG TRIỂN KHAI** (Synced with `request.md`)

---

# 1. Mục tiêu dự án

Xây dựng pipeline huấn luyện, đánh giá và so sánh hai mô hình object detection hiện đại dựa trên kiến trúc Transformer:

1. **RT-DETR** (Real-Time DEtection TRansformer từ Ultralytics / Baidu)
2. **RF-DETR** (Roboflow Detection Transformer)

Môi trường và tài nguyên:
- **Môi trường tính toán**: Modal GPU (A10G / L4 / L40S / A100).
- **Lưu trữ dữ liệu & checkpoint**: Modal Volume (`/vol` -> `food-detection-training`).
- **Dataset**: `completed-project` (Version 1) được quản lý trên Roboflow: [https://app.roboflow.com/nckhcict2025/completed-project/1](https://app.roboflow.com/nckhcict2025/completed-project/1)
- **Bài toán**: Nhận dạng đối tượng thực phẩm, nguyên liệu, rau củ quả, đồ uống trong ảnh.

Kết quả cuối cùng cần bàn giao:
- Pipeline tự động download và chuẩn hóa dataset từ Roboflow.
- Pipeline huấn luyện RT-DETR và RF-DETR chạy trên Modal GPU trong đúng **50 epochs**.
- Cơ chế theo dõi và ghi nhận chi tiết theo từng epoch: **Tổng Train Loss, Class Loss, Box Loss, GIoU Loss, Learning Rate, và Thời gian thực thi (giây)**.
- Lưu trữ checkpoint chuẩn: **best.pt / best.pth** (dựa trên mAP@50-95 cao nhất trên tập Validation) và **last.pt / last.pth** (epoch 50).
- Báo cáo kết quả đánh giá chi tiết: **Precision, Recall, mAP50, mAP50-95, độ chính xác từng lớp trên tập Test và tổng thể**.
- Script inference kiểm thử kết quả trên ảnh thực tế.
- Khả năng resume nếu quá trình train trên Modal bị gián đoạn.

---

# 2. Yêu cầu gốc (Trích xuất từ `request.md`)

## 2.1. Nguồn yêu cầu
File: `D:\NCKH\Train_models\request.md`

## 2.2. Bảng trích xuất yêu cầu chi tiết

| STT | Yêu cầu | Chi tiết kỹ thuật | Trạng thái |
|---|---|---|---|
| 1 | **Mô hình** | Huấn luyện 2 mô hình: RT-DETR và RF-DETR | SẴN SÀNG |
| 2 | **Môi trường chạy** | Chạy trên Modal GPU, lưu trữ kết quả trên Modal Volume Storage | SẴN SÀNG |
| 3 | **Dataset URL** | `https://app.roboflow.com/nckhcict2025/completed-project/1`<br>Workspace: `nckhcict2025`<br>Project: `completed-project`<br>Version: `1` | ĐÃ XÁC ĐỊNH |
| 4 | **Số lượng Epoch** | **50 Epochs** cho cả hai mô hình | ĐÃ THIẾT LẬP |
| 5 | **Tracking Loss & Params** | Ghi log từng epoch: `Epoch`, `Train loss`, `Class loss`, `Box loss`, `GIoU`, `Learning rate`, `Runtime` | ĐÃ CẤU HÌNH |
| 6 | **Lưu Checkpoints** | - `best.pt` / `best.pth` dựa theo `mAP@50-95` cao nhất trên tập Validation<br>- `last.pt` / `last.pth` (epoch 50) | ĐÃ CẤU HÌNH |
| 7 | **Đánh giá & Báo cáo** | - Độ chính xác: Precision, Recall, mAP50, mAP50-95<br>- Độ chính xác của từng lớp (dựa trên tập Test)<br>- Độ chính xác tổng thể trên toàn bộ tập dữ liệu | ĐÃ CẤU HÌNH |
| 8 | **Lý thuyết tham khảo** | Box Loss (tổng mức độ sai lệch khung) và GIoU (Generalized IoU giải quyết bài toán khoảng cách và kéo khung) | ĐÃ TÍCH HỢP |

---

# 3. Nguyên tắc thiết kế & Tái lập

1. **Cùng tập dữ liệu logic**: RT-DETR và RF-DETR sử dụng cùng một phiên bản dataset (Version 1 từ Roboflow), cùng train/val/test split, cùng danh sách class ID.
2. **Khóa Test Set**: Tập test chỉ dùng để đánh giá sau khi hoàn tất training và validation, không dùng test set để điều chỉnh hyperparameter hay early stopping.
3. **Tracking & Tái lập (Reproducibility)**:
   - Cố định random seed (mặc định: 42).
   - Lưu lại toàn bộ siêu tham số: `image_size`, `batch_size`, `learning_rate`, `weight_decay`, `optimizer`, `loss components`.

---

# 4. Thiết kế Dataset & Roboflow Integration

## 4.1. Thông tin Dataset
- **Roboflow Workspace**: `nckhcict2025`
- **Project**: `completed-project`
- **Version**: `5`
- **URL**: [https://app.roboflow.com/nckhcict2025/completed-project/5](https://app.roboflow.com/nckhcict2025/completed-project/5)

## 4.2. Định dạng Export
- **RT-DETR (Ultralytics)**: Hỗ trợ định dạng `yolov8` / `yolov11` (YOLO format với `data.yaml`).
- **RF-DETR**: Hỗ trợ định dạng `coco` hoặc `yolo`.
- **Chiến lược**:
  - Script download sẽ tải dataset theo format tương ứng vào thư mục lưu trữ persistent trên Modal Volume:
    - `/vol/datasets/completed-project-5/yolo/`
    - `/vol/datasets/completed-project-5/coco/`

## 4.3. Quản lý Secret
- Roboflow API Key được nạp thông qua biến môi trường `ROBOFLOW_API_KEY` hoặc cấu hình qua Modal Secret `roboflow-secret`.

---

# 5. Cấu trúc mã nguồn dự án

```text
Train_models/
├── request.md                  # File yêu cầu ban đầu của đề tài
├── TRAINING_PLAN.md            # File kế hoạch và nhật ký trạng thái
├── requirements.txt            # Danh sách thư viện Python
│
├── configs/                    # File cấu hình tham số
│   ├── dataset.yaml            # Cấu hình dataset (Roboflow URL, paths, classes)
│   ├── rtdetr.yaml             # Cấu hình train RT-DETR (50 epochs, imgsz=640, loss tracking)
│   └── rfdetr.yaml             # Cấu hình train RF-DETR (50 epochs, imgsz=640, loss tracking)
│
├── src/                        # Mã nguồn chính
│   ├── data/
│   │   ├── roboflow_download.py # Tải dữ liệu từ Roboflow API
│   │   └── validate_dataset.py  # Kiểm tra tính toàn vẹn và phân bố nhãn
│   ├── common/
│   │   ├── epoch_logger.py      # Logger ghi nhận: Epoch, Train Loss, Class Loss, Box Loss, GIoU, LR, Time
│   │   └── metrics_reporter.py  # Tính toán Precision, Recall, mAP50, mAP50-95, per-class report
│   ├── rtdetr/
│   │   ├── train.py             # Huấn luyện RT-DETR với custom loss callbacks
│   │   └── evaluate.py          # Đánh giá RT-DETR trên tập test
│   └── rfdetr/
│       ├── train.py             # Huấn luyện RF-DETR với tracking loss và checkpoints
│       └── evaluate.py          # Đánh giá RF-DETR trên tập test
│
└── modal_app/                  # Triển khai trên nền tảng Modal GPU
    ├── app.py                  # Định nghĩa Modal App, Image, GPU, Volume và Secrets
    └── pipeline.py             # CLI runner điều phối các tác vụ trên Cloud GPU
```

---

# 6. Kiến trúc Modal GPU & Lưu trữ Volume

## 6.1. Modal Volume
- **Tên Volume**: `food-detection-training`
- **Mount Path**: `/vol`
- **Cấu trúc lưu trữ trên Volume**:
  ```text
  /vol/
  ├── datasets/
  │   └── completed-project-1/
  │       ├── yolo/
  │       └── coco/
  ├── checkpoints/
  │   ├── rtdetr/
  │   │   ├── best.pt           # mAP@50-95 cao nhất
  │   │   └── last.pt           # Epoch 50
  │   └── rfdetr/
  │       ├── best.pth          # mAP@50-95 cao nhất
  │       └── last.pth          # Epoch 50
  ├── logs/
  │   ├── rtdetr_epoch_logs.csv # Bảng log chi tiết từng epoch
  │   ├── rfdetr_epoch_logs.csv # Bảng log chi tiết từng epoch
  │   └── comparison_report.md  # Báo cáo so sánh 2 mô hình
  └── outputs/
      ├── rtdetr_eval/
      └── rfdetr_eval/
  ```

## 6.2. Cấu hình GPU
- Mặc định: `modal.gpu.A10G` hoặc `modal.gpu.L4` (tối ưu chi phí và VRAM 24GB).
- Nâng cấp nếu cần batch size lớn hơn: `modal.gpu.A100` hoặc `modal.gpu.L40S`.

---

# 7. Thiết kế Chi Tiết Mô Hình & Loss Tracking

## 7.1. RT-DETR (Ultralytics)
- **Model variant**: `rtdetr-l.pt` (hoặc `rtdetr-x.pt` nếu cần dung lượng lớn hơn).
- **Epochs**: **50**.
- **Loss Components**:
  - `class_loss`: BCE / Focal Loss cho phân loại đối tượng.
  - `box_loss`: L1 Bounding Box Regression Loss.
  - `giou_loss`: Generalized IoU Loss đo độ tương đồng hình học và định vị khung.
  - `total_loss = w_cls * class_loss + w_box * box_loss + w_giou * giou_loss`.
- **Validation Metric**: Giám sát `metrics/mAP50-95(B)` để lưu `best.pt`.

## 7.2. RF-DETR (Roboflow DETR)
- **Model variant**: `RFDETRMedium` / `RFDETRBase`.
- **Epochs**: **50**.
- **Loss Components**:
  - `loss_vfl` (Classification / Varifocal Loss).
  - `loss_bbox` (L1 Bounding Box Regression).
  - `loss_giou` (GIoU Loss).
  - `total_loss` tổng hợp.
- **Validation Metric**: Giám sát `mAP_50_95` trên Validation COCO evaluator để lưu `best.pth`.

---

# 8. Mẫu Báo Cáo Kết Quả (Format Chuẩn)

### 8.1. Bảng Log Theo Từng Epoch (Epoch-by-Epoch Training Log)

| Epoch | Train Loss | Class Loss | Box Loss | GIoU Loss | Learning Rate | Epoch Time (s) |
|---|---|---|---|---|---|---|
| 1/50 | ... | ... | ... | ... | ... | ... |
| 2/50 | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |
| 50/50 | ... | ... | ... | ... | ... | ... |

### 8.2. Bảng Đánh Giá Độ Chính Xác Từng Lớp (Per-Class Accuracy on Test Set)

| Class ID | Class Name | Instances (Test) | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| 0 | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |
| **All** | **Tổng thể** | **...** | **...** | **...** | **...** | **...** |

### 8.3. Bảng So Sánh Hai Mô Hình (Model Comparison Scorecard)

| Tiêu chí | RT-DETR | RF-DETR | Chênh lệch (Difference) |
|---|---|---|---|
| **mAP@50-95 (Val Best)** | ... | ... | ... |
| **mAP@50-95 (Test)** | ... | ... | ... |
| **mAP@50 (Test)** | ... | ... | ... |
| **Precision (Test)** | ... | ... | ... |
| **Recall (Test)** | ... | ... | ... |
| **Model Checkpoint Size** | ... | ... | ... |
| **Tổng thời gian train (50 epochs)** | ... | ... | ... |
| **Inference Latency (ms/ảnh)** | ... | ... | ... |

---

# 9. Phụ Lục: Lý Thuyết Box Loss và GIoU

## 9.1. Box Loss (Tổng mức độ sai lệch khung)
- **Bản chất**: Sai số tổng hợp đo khoảng cách giữa tọa độ tâm, chiều rộng và chiều cao của khung dự đoán so với ground truth (thường sử dụng khoảng cách chuẩn hóa L1 hoặc Smooth L1).
- **Hạn chế**: Khi hai khung hoàn toàn không giao nhau (IoU = 0), gradient của hàm khoảng cách đơn thuần không thể hiện được hướng di chuyển tối ưu trong không gian 2D, khiến quá trình hội tụ bị chậm.

## 9.2. GIoU - Generalized Intersection over Union
- **Công thức**:
  $$\text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}$$
  *Trong đó $A$ là bounding box dự đoán, $B$ là ground truth, $C$ là bounding box lồi nhỏ nhất (convex hull) bao trùm cả $A$ và $B$.*
- **Tác dụng**:
  1. Khi $A$ và $B$ không giao nhau ($\text{IoU} = 0$), $\text{GIoU} < 0$. Giá trị này phản ánh diện tích vùng trống dư thừa cần thu hẹp.
  2. Tạo ra "lực kéo" gradient hướng khung dự đoán dịch chuyển về phía đối tượng thực tế ngay cả khi hai khung ở cách xa nhau.
  3. Xử lý triệt để bài toán đối tượng thực phẩm nằm rời rạc hoặc xếp chồng lấn lên nhau trong ảnh.

---

# 10. Nhật Ký Tiến Độ (Progress Log)

## 2026-08-28
- [x] Đọc và trích xuất 100% yêu cầu từ `request.md`.
- [x] Đồng bộ thông tin dataset: `https://app.roboflow.com/nckhcict2025/completed-project/1`.
- [x] Thiết lập số epoch = **50 epochs** cho cả RT-DETR và RF-DETR.
- [x] Thiết kế cấu trúc lưu trữ và bóc tách loss: Train Loss, Class Loss, Box Loss, GIoU, LR, Runtime.
- [x] Thiết lập quy chuẩn lưu checkpoint: `best` (mAP50-95) và `last` (epoch 50).
- [x] Cập nhật toàn diện `TRAINING_PLAN.md`.
- [x] Xây dựng bộ mã nguồn Python & cấu hình (`configs/`, `src/`, `modal_app/`, `requirements.txt`).
- [x] Kiểm tra cú pháp và tính sẵn sàng của toàn bộ modules (`python -m py_compile` pass).

---

# 11. Quyết Định Kỹ Thuật (Decision Log)

- **[DECISION-001] (2026-08-28)**: Sử dụng Modal Volume `food-detection-training` gắn tại `/vol` để lưu trữ tập trung dataset, checkpoints và logs, đảm bảo không bị mất dữ liệu khi container Modal kết thúc.
- **[DECISION-002] (2026-08-28)**: Bóc tách loss components thông qua custom callback/hooks trong quá trình huấn luyện của cả hai framework Ultralytics và RF-DETR nhằm đáp ứng chính xác bảng log gồm 6 cột: `Epoch`, `Train loss`, `Class loss`, `Box loss`, `GIoU`, `Learning rate`.
- **[DECISION-003] (2026-08-28)**: Cố định 50 epochs và lấy checkpoint `best` theo metric `mAP@50-95` trên tập Validation để bảo đảm tính chuẩn xác cao nhất cho việc định vị bounding box.
- **[DECISION-004] (2026-08-28)**: Bổ sung cơ chế bảo vệ khi Timeout/SIGTERM: Tự động bắt Signal `SIGTERM`/`SIGINT`, khối `try...finally` và `on_fit_epoch_end` đảm bảo `best.pt`, `last.pt`, log CSV luôn được lưu và gọi `volume.commit()` ngay lập tức sau mỗi epoch. Khi chạy lại, `load_existing_logs()` tự động đọc log cũ để không bị ghi đè.
- **[DECISION-005] (2026-08-28)**: Áp dụng cơ chế Atomic Checkpoint (`.tmp` -> rename) và cấu trúc phân lập Experiment (`/vol/experiments/<model>/<experiment_id>/`) để chống ghi đè dữ liệu và chống hỏng file khi crash giữa chừng.

---

# 12. Error Log & Audit Rủi Ro Tiềm Ẩn (Potential Issues)

```text
[AUDIT-001]
Component: Checkpoint Strategy
Problem: Không có cơ chế ghi Atomic Checkpoint (.tmp -> rename), rủi ro hỏng file nếu container chết đúng lúc đang ghi đĩa.
Status: POTENTIAL ISSUE

[AUDIT-002]
Component: Experiment Isolation
Problem: Đường dẫn output hiện đang tĩnh (/vol/checkpoints/rtdetr), có thể ghi đè nếu chạy nhiều experiment khác nhau.
Status: POTENTIAL ISSUE

[AUDIT-003]
Component: Auto-Resume Validation
Problem: Chưa xác thực metadata (model architecture, dataset version, num_classes) trước khi resume, nguy cơ resume nhầm model khác.
Status: POTENTIAL ISSUE

[AUDIT-004]
Component: GPU Memory (Validation OOM)
Problem: Validation batch size mặc định của Ultralytics có thể gấp đôi train batch size, dễ gây CUDA OOM ở epoch validation đầu tiên.
Status: POTENTIAL ISSUE

[AUDIT-005]
Component: Dataloader Worker & Shared Memory
Problem: num_workers=4 trong container Modal có thể làm cạn /dev/shm nếu kích thước ảnh hoặc batch lớn.
Status: POTENTIAL ISSUE

[AUDIT-006]
Component: Training Stability (NaN Loss)
Problem: Chưa có cơ chế phát hiện sớm NaN/Inf loss để dừng an toàn và lưu diagnostic batch.
Status: POTENTIAL ISSUE

[AUDIT-007]
Component: Dependencies Pinning
Problem: requirements.txt sử dụng version mở (>=), nguy cơ xung đột khi thư viện upstream cập nhật breaking changes.
Status: POTENTIAL ISSUE
```

---

# 13. Pre-Flight Checklist (Kiểm tra trước khi bấm Train)

- [ ] **1. Roboflow Dataset**: Version 1 từ `nckhcict2025/completed-project` đã tải và validate cấu trúc ảnh/nhãn thành công.
- [ ] **2. Modal Secret**: Secret `roboflow-secret` hoặc token `ROBOFLOW_API_KEY` đã được thiết lập.
- [ ] **3. Modal Volume**: Volume `food-detection-training` được mount chính xác tại `/vol` và có quyền ghi.
- [ ] **4. Experiment ID**: Experiment ID là duy nhất (ví dụ: `rtdetr_50ep_v1_20260828_01`).
- [ ] **5. GPU & VRAM**: Chọn đúng GPU `A10G` (24GB VRAM) hoặc `L4` đảm bảo không OOM với batch size 16 (RT-DETR) và 8 (RF-DETR).
- [ ] **6. Dataloader Workers**: `num_workers` thiết lập ở mức an toàn (2 hoặc 4).
- [ ] **7. Atomic Checkpoint**: Đã bật cơ chế ghi tạm `.tmp` trước khi đổi tên thành `.pt`/`.pth`.
- [ ] **8. Timeout Budget**: Thiết lập timeout cho Modal Function từ `43200s` (12 giờ) trở lên.
- [ ] **9. Checkpoint Metadata**: Kiểm tra tính khớp nối của `num_classes` và `dataset_version` khi thực hiện resume.
- [ ] **10. Smoke Test**: Chạy thử 1 epoch smoke test trên subset để kiểm tra toàn bộ luồng trước khi chạy chính thức 50 epochs.

