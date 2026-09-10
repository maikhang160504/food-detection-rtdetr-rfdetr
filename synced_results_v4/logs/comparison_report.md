# Báo Cáo So Sánh Đối Đầu Toàn Diện: RT-DETR vs RF-DETR (v4)

## 1. So Sánh Độ Chính Xác Trên Tập Test (Accuracy Benchmark)

| Tiêu chí         |   RT-DETR |   RF-DETR |   Chênh lệch (RF vs RT) |
|:-----------------|----------:|----------:|------------------------:|
| mAP@50-95 (Test) |    0.8746 |    0.862  |                 -0.0125 |
| mAP@50 (Test)    |    0.9825 |    0.9679 |                 -0.0146 |
| Precision (Test) |    0.977  |    0.9752 |                 -0.0018 |
| Recall (Test)    |    0.969  |    0.9744 |                  0.0054 |
| F1-Score (Test)  |    0.973  |    0.9748 |                  0.0018 |

## 2. So Sánh Hiệu Năng Phần Cứng & Tốc Độ Trên GPU A100 (Hardware Benchmark)

| Chỉ số phần cứng           | RT-DETR      | RF-DETR     | Nhận xét                             |
|:---------------------------|:-------------|:------------|:-------------------------------------|
| Parameters (Số tham số)    | 32.87 M      | 33.6 M      | Cân xứng tương đương (~32M)          |
| GFLOPs (tại 640x640)       | 110.0 GFLOPs | 2.87 GFLOPs | RF-DETR tối ưu hơn về phép tính      |
| Độ trễ Latency (@ Batch=1) | 7.50 ms      | 8.20 ms     | Đo bằng Warmup + CUDA Sync trên A100 |
| Tốc độ Throughput (FPS)    | 133.3 FPS    | 121.9 FPS   | Khả năng xử lý thời gian thực        |

## 3. Trực Quan Hóa Ma Trận Nhầm Lẫn (Confusion Matrix)

- **RT-DETR Confusion Matrix**: `outputs/rtdetr_eval/confusion_matrix.png` & `confusion_matrix_normalized.png`
- **RF-DETR Confusion Matrix**: `outputs/rfdetr_eval/confusion_matrix.png` & `confusion_matrix_normalized.png`

