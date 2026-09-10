# Báo Cáo Đánh Giá Mô Hình: RT-DETR (Phiên bản v4)

## 1. Độ Chính Xác Tổng Thể Trên Tập Test (Chuẩn COCO)

- **Precision (P)**: 0.9770
- **Recall (R)**: 0.9690
- **F1-Score**: 0.9730
- **mAP@50**: 0.9825
- **mAP@50-95**: 0.8746

## 2. Hiệu Năng Phần Cứng & Tốc Độ Suy Luận (Inference Benchmark)

| Chỉ số phần cứng                | Giá trị               |
|:--------------------------------|:----------------------|
| Thiết bị GPU                    | NVIDIA A100-SXM4-40GB |
| Kích thước đầu vào (Resolution) | 640x640               |
| Batch Size kiểm thử             | 1                     |
| Số lượng tham số (Parameters)   | 32.87 M               |
| Độ phức tạp tính toán (GFLOPs)  | 109.32 GFLOPs         |
| Độ trễ suy luận (GPU Latency)   | 39.57 ms              |
| Tốc độ xử lý (Throughput)       | 25.3 FPS              |

## 3. Độ Chính Xác Từng Lớp Trên Tập Test (Per-Class Accuracy)

|   Class ID | Class Name   |   Instances |   Precision |   Recall |   mAP@50 |   mAP@50-95 |
|-----------:|:-------------|------------:|------------:|---------:|---------:|------------:|
|          0 | beef         |         331 |      0.9878 |   0.9772 |   0.99   |      0.9049 |
|          1 | bellpepper   |          83 |      0.9537 |   0.9639 |   0.9733 |      0.9275 |
|          2 | bittergourd  |          50 |      0.9966 |   1      |   0.995  |      0.9529 |
|          3 | bottlegourd  |         140 |      0.9848 |   0.9263 |   0.9668 |      0.8576 |
|          4 | broccoli     |          68 |      0.9845 |   0.9853 |   0.9819 |      0.9435 |
|          5 | cabbage      |          79 |      0.9732 |   0.9195 |   0.9796 |      0.895  |
|          6 | carrot       |          74 |      0.978  |   0.9865 |   0.9922 |      0.8645 |
|          7 | cauliflower  |         109 |      0.9529 |   1      |   0.9857 |      0.9188 |
|          8 | chayote      |          57 |      0.9969 |   1      |   0.995  |      0.9565 |
|          9 | chicken      |          50 |      0.9659 |   0.9    |   0.9563 |      0.6821 |
|         10 | chickenegg   |         223 |      0.9604 |   0.9865 |   0.9928 |      0.9011 |
|         11 | chickenleg   |         531 |      0.988  |   0.9849 |   0.9926 |      0.8485 |
|         12 | chickenwin   |         151 |      0.9983 |   1      |   0.995  |      0.8871 |
|         13 | corn         |         134 |      0.9419 |   0.903  |   0.9539 |      0.8003 |
|         14 | cucumber     |         197 |      0.941  |   0.8901 |   0.9409 |      0.8372 |
|         15 | duckegg      |          99 |      0.9892 |   0.9697 |   0.9801 |      0.9085 |
|         16 | eggplant     |          45 |      0.9924 |   1      |   0.995  |      0.9127 |
|         17 | garlic       |         179 |      0.9887 |   0.9761 |   0.9869 |      0.8551 |
|         18 | ginger       |          64 |      1      |   0.9983 |   0.995  |      0.9137 |
|         19 | jicama       |         109 |      0.9586 |   0.8991 |   0.9739 |      0.8631 |
|         20 | okra         |          92 |      0.9678 |   0.9811 |   0.9757 |      0.7158 |
|         21 | onion        |         126 |      0.9529 |   0.9603 |   0.9724 |      0.8191 |
|         22 | pork         |          79 |      0.9748 |   0.9794 |   0.9842 |      0.8684 |
|         23 | potato       |          56 |      0.9861 |   1      |   0.995  |      0.9288 |
|         24 | pumpkin      |         104 |      0.9903 |   0.9864 |   0.9949 |      0.9369 |
|         25 | radish       |         159 |      0.9848 |   0.9811 |   0.9944 |      0.8817 |
|         26 | scallion     |          55 |      0.9637 |   0.966  |   0.9868 |      0.7973 |
|         27 | shrimp       |          48 |      0.9971 |   1      |   0.995  |      0.913  |
|         28 | spongegourd  |         143 |      0.9919 |   0.986  |   0.9902 |      0.895  |
|         29 | sweetpotato  |         100 |      0.9483 |   0.9179 |   0.9435 |      0.8279 |
|         30 | tofu         |          44 |      0.9967 |   1      |   0.995  |      0.9085 |
|         31 | tomato       |         132 |      0.9774 |   0.9849 |   0.9922 |      0.8628 |

## 4. Ma Trận Nhầm Lẫn (Confusion Matrix)

- **confusion_matrix.png**: `confusion_matrix.png`
- **confusion_matrix_normalized.png**: `confusion_matrix_normalized.png`
- **confusion_matrix.csv**: `confusion_matrix.csv`
- **confusion_matrix_normalized.csv**: `confusion_matrix_normalized.csv`
- **per_class_metrics.csv**: `per_class_metrics.csv`

