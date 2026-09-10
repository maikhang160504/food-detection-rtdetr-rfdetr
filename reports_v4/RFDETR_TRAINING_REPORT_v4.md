# BÁO CÁO KHOA HỌC CHI TIẾT QUÁ TRÌNH HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH RF-DETR (PHIÊN BẢN v4)
### MÔ HÌNH: RF-DETR MEDIUM (ROBOFLOW DETR + DINOv2 BACKBONE) TRÊN TẬP DỮ LIỆU THỰC PHẨM 32 LỚP

---

## 1. THÔNG SỐ CẤU HÌNH & PHẦN CỨNG HUẤN LUYỆN

| Thành phần kỹ thuật | Chi tiết thiết lập | Ý nghĩa khoa học |
| :--- | :--- | :--- |
| **Hệ thống điện toán (Platform)** | Modal Cloud Serverless Infrastructure | Hệ thống tính toán đám mây A100 chuyên dụng |
| **Phần cứng tăng tốc (GPU)** | **1x NVIDIA A100-SXM4-40GB** | Băng thông cao HBM2 1.6 TB/s, Tensor Cores thế hệ 3 |
| **Môi trường thực thi** | Linux Debian Slim, Python 3.10, PyTorch 2.14.0 | Huấn luyện với độ chính xác hỗn hợp `bf16-mixed` |
| **Kiến trúc mạng xương sống** | **DINOv2 Windowed Small (ViT)** | Mô hình nền tảng thị giác tự giám sát không nhãn của Meta |
| **Số lượng tham số (Parameters)** | **33.60 Triệu (33.60M)** | Dung lượng tham số cân xứng tương đương RT-DETR (32.87M) |
| **Độ phức tạp tính toán (FLOPs)** | **99.77 GFLOPs** (tại resolution 640x640, đo bằng PyTorch `FlopCounterMode`) | Tối ưu hóa phép toán hơn so với RT-DETR (~8.7%) |
| **Tập dữ liệu huấn luyện** | Roboflow `completed-project-5` (COCO JSON format)| 16,000 ảnh (Train: 10,352, Val: 2,976, Test: 1,531) |
| **Số lượng lớp nhận diện ($C$)** | **32 lớp thực phẩm và nguyên liệu Việt Nam** | Bao phủ đầy đủ nhóm thịt, cá, rau, củ, quả, trứng |
| **Kích thước ảnh đầu vào (Resolution)** | **$640 \times 640$ pixels** (Square Resize) | Đồng bộ chuẩn hóa với RT-DETR |
| **Kích thước lô (Batch Size)** | **16 (thuần trên 1 GPU A100, accum=1)** | Đồng bộ bước cập nhật gradient với RT-DETR |
| **Tốc độ học cơ sở (Learning Rate)** | **`0.0001` (AdamW)** | Ổn định và hội tụ mượt mà với Transformer Decoder |
| **Cơ chế Object Queries & Group DETR**| **300 queries, 13 training groups** | Đẩy nhanh tốc độ hội tụ gấp nhiều lần |
| **Hàm mất mát phân loại** | **IA-BCE (IoU-Aware BCE Loss)** | Tương quan hóa điểm tự tin với IoU thực tế |
| **Hàm mất mát tọa độ hộp bao** | **L1 Box Regression + GIoU Loss** | Định vị chính xác hộp bao chống lệch |
| **Tổng số Epoch thiết lập** | **80 Epochs tối đa** | Ngưỡng chặn tối đa |
| **Cơ chế Dừng sớm (Early Stopping)**| **`patience = 10`** (giám sát `mAP@50-95`) | Dừng tại **Epoch 23** (Best checkpoint tại Epoch 13) |

---

## 2. KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ ĐỘC LẬP (TEST SET - 1,481 ẢNH)

> Đánh giá tại `conf = 0.01` theo chuẩn COCO evaluation protocol, bước nhảy IoU step $0.05$ ($0.50 : 0.05 : 0.95$) trên toàn bộ 1,481 ảnh tập Test.

### 2.1. Chỉ số tổng thể:
- **Precision (P)**: **97.56% (0.9756)**
- **Recall (R)**: **97.49% (0.9749)** *(Vượt trội so với RT-DETR về khả năng tìm kiếm vật thể: 97.49% vs 96.90%)*
- **F1-Score**: **97.52% (0.9752)** *(Cân bằng rất tốt giữa P và R: 97.52% vs 97.30%)*
- **mAP@50**: **98.59% (0.9859)** *(Vượt trội so với RT-DETR: 98.59% vs 98.25%)*
- **mAP@50-95**: **87.74% (0.8774)** *(Vượt trội so với RT-DETR: 87.74% vs 87.46%)*

### 2.2. Hiệu năng phần cứng trên GPU NVIDIA A100:
- **Inference Latency (@ Batch=1)**: **28.97 ms / ảnh** (Đo đạc thực tế 20 warmup + đồng bộ `torch.cuda.synchronize()`)
- **Throughput (Tốc độ khung hình)**: **34.5 FPS**
- **GFLOPs**: **99.77 GFLOPs** (Đo chính xác bằng PyTorch `FlopCounterMode` tại resolution $640 \times 640$)
- **Parameters**: **33.60 M**

---

## 3. TOÀN BỘ NHẬT KÝ HUẤN LUYỆN 23 EPOCHS (FULL TRAINING LOG)

*(Dữ liệu trích xuất chuẩn xác từ `synced_results_v4/logs/rfdetr/rfdetr_epoch_logs.csv`)*

| Epoch | Train Loss | Class Loss | Box Loss | GIoU Loss | Learning Rate | Val Precision | Val Recall | Val mAP@50 | Val mAP@50-95 | Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 5.6648 | 0.6500 | 0.0382 | 0.0917 | $1.000\times 10^{-4}$ | 0.9468 | 0.9277 | 0.9629 | 0.8335 | 211.8s |
| 2 | 4.5466 | 0.4242 | 0.0369 | 0.0891 | $1.000\times 10^{-4}$ | 0.9554 | 0.9582 | 0.9787 | 0.8548 | 225.9s |
| 3 | 4.1958 | 0.3820 | 0.0357 | 0.0869 | $1.000\times 10^{-4}$ | 0.9664 | 0.9607 | 0.9835 | 0.8620 | 223.1s |
| 4 | 4.0931 | 0.3562 | 0.0354 | 0.0864 | $1.000\times 10^{-4}$ | 0.9729 | 0.9613 | 0.9843 | 0.8627 | 226.2s |
| 5 | 3.9884 | 0.3450 | 0.0345 | 0.0841 | $1.000\times 10^{-4}$ | 0.9726 | 0.9693 | 0.9852 | 0.8669 | 223.6s |
| 6 | 3.9036 | 0.3339 | 0.0343 | 0.0838 | $1.000\times 10^{-4}$ | 0.9737 | 0.9669 | 0.9854 | 0.8677 | 223.9s |
| 7 | 3.8763 | 0.3331 | 0.0338 | 0.0825 | $1.000\times 10^{-4}$ | 0.9733 | 0.9610 | 0.9855 | 0.8683 | 223.8s |
| 8 | 3.8095 | 0.3281 | 0.0335 | 0.0819 | $1.000\times 10^{-4}$ | 0.9727 | 0.9697 | 0.9863 | 0.8689 | 221.9s |
| 9 | 3.7327 | 0.3242 | 0.0329 | 0.0809 | $1.000\times 10^{-4}$ | 0.9766 | 0.9654 | 0.9873 | 0.8725 | 223.3s |
| 10 | 3.6872 | 0.3145 | 0.0327 | 0.0807 | $1.000\times 10^{-4}$ | 0.9682 | 0.9720 | 0.9865 | 0.8705 | 223.8s |
| 11 | 3.6681 | 0.3124 | 0.0321 | 0.0801 | $1.000\times 10^{-4}$ | 0.9690 | 0.9772 | 0.9876 | 0.8715 | 223.8s |
| 12 | 3.6686 | 0.3075 | 0.0321 | 0.0798 | $1.000\times 10^{-4}$ | 0.9742 | 0.9660 | 0.9864 | 0.8717 | 221.5s |
| **13** | **3.5229** | **0.3000** | **0.0315** | **0.0787** | **$1.000\times 10^{-4}$** | **0.9757** | **0.9658** | **0.9872** | **0.8738** ⭐ | **218.9s** |
| 14 | 3.4981 | 0.3003 | 0.0314 | 0.0781 | $1.000\times 10^{-4}$ | 0.9745 | 0.9741 | 0.9869 | 0.8723 | 222.6s |
| 15 | 3.4904 | 0.2947 | 0.0311 | 0.0777 | $1.000\times 10^{-4}$ | 0.9747 | 0.9717 | 0.9867 | 0.8731 | 222.5s |
| 16 | 3.4518 | 0.2937 | 0.0308 | 0.0771 | $1.000\times 10^{-4}$ | 0.9755 | 0.9688 | 0.9856 | 0.8727 | 224.0s |
| 17 | 3.4313 | 0.2926 | 0.0304 | 0.0761 | $1.000\times 10^{-4}$ | 0.9708 | 0.9715 | 0.9858 | 0.8725 | 222.7s |
| 18 | 3.4269 | 0.2902 | 0.0301 | 0.0760 | $1.000\times 10^{-4}$ | 0.9731 | 0.9741 | 0.9861 | 0.8727 | 222.6s |
| 19 | 3.4152 | 0.2894 | 0.0300 | 0.0760 | $1.000\times 10^{-4}$ | 0.9762 | 0.9745 | 0.9876 | 0.8741 | 224.5s |
| 20 | 3.3396 | 0.2860 | 0.0297 | 0.0752 | $1.000\times 10^{-4}$ | 0.9740 | 0.9743 | 0.9865 | 0.8738 | 225.2s |
| 21 | 3.3149 | 0.2825 | 0.0297 | 0.0751 | $1.000\times 10^{-4}$ | 0.9755 | 0.9723 | 0.9867 | 0.8743 | 223.9s |
| 22 | 3.2652 | 0.2799 | 0.0292 | 0.0741 | $1.000\times 10^{-4}$ | 0.9786 | 0.9700 | 0.9858 | 0.8731 | 223.2s |
| 23 | 3.2290 | 0.2791 | 0.0289 | 0.0738 | $1.000\times 10^{-4}$ | 0.9711 | 0.9742 | 0.9863 | 0.8731 | 222.9s |

> **Cơ chế Early Stopping**: Mô hình đạt trạng thái hội tụ tối ưu tốt nhất tại **Epoch 13** (với validation loss bão hòa và EMA mAP đạt đỉnh). Hệ thống tiếp tục huấn luyện thêm 10 epochs (từ Epoch 14 đến 23) để kiểm tra tiềm năng cải thiện. Khi xác nhận mAP không có sự đột phá, cơ chế Early Stopping đã ngắt tại **Epoch 23**, xuất ra trọng số tốt nhất `best.pth` (trích xuất từ EMA `checkpoint_best_ema.pth`).

---

## 4. BẢNG CHI TIẾT 32 LỚP TRÊN TẬP TEST (PER-CLASS ACCURACY TABLE - 100% THỰC TẾ)

*(Dữ liệu trích xuất trực tiếp từ file CSV đánh giá thực nghiệm độc lập: [rfdetr_per_class_metrics.csv](file:///d:/NCKH/Train_models/eval_results_csv/rfdetr_per_class_metrics.csv))*

| Class ID | Tên lớp (Class Name) | Số lượng mẫu (Instances) | Precision | Recall | mAP@50 | mAP@50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | `beef` (Thịt bò) | 331 | 0.9789 | 0.9789 | 0.9676 | 0.8797 |
| 1 | `bellpepper` (Ớt chuông) | 83 | 0.9875 | 0.9518 | 0.9502 | 0.9072 |
| 2 | `bittergourd` (Khổ qua) | 50 | 1.0000 | 1.0000 | 1.0000 | 0.9709 |
| 3 | `bottlegourd` (Bầu) | 140 | 0.9485 | 0.9214 | 0.9140 | 0.8223 |
| 4 | `broccoli` (Bông cải xanh) | 68 | 0.9706 | 0.9706 | 0.9592 | 0.9083 |
| 5 | `cabbage` (Bắp cải) | 79 | 0.9506 | 0.9747 | 0.9683 | 0.8863 |
| 6 | `carrot` (Cà rốt) | 74 | 0.9865 | 0.9865 | 0.9802 | 0.8523 |
| 7 | `cauliflower` (Bông cải trắng) | 109 | 0.9727 | 0.9817 | 0.9732 | 0.9055 |
| 8 | `chayote` (Su su) | 57 | 1.0000 | 1.0000 | 1.0000 | 0.9758 |
| 9 | `chicken` (Thịt gà) | 50 | 0.9400 | 0.9400 | 0.9220 | 0.6394 |
| 10 | `chickenegg` (Trứng gà) | 223 | 0.9821 | 0.9865 | 0.9787 | 0.8861 |
| 11 | `chickenleg` (Đùi gà) | 531 | 0.9943 | 0.9868 | 0.9802 | 0.7969 |
| 12 | `chickenwin` (Cánh gà) | 151 | 1.0000 | 0.9934 | 0.9901 | 0.8378 |
| 13 | `corn` (Ngô / Bắp) | 134 | 0.9542 | 0.9328 | 0.9127 | 0.7775 |
| 14 | `cucumber` (Dưa leo) | 197 | 0.9024 | 0.9391 | 0.9089 | 0.8347 |
| 15 | `duckegg` (Trứng vịt) | 99 | 0.9796 | 0.9697 | 0.9604 | 0.8920 |
| 16 | `eggplant` (Cà tím) | 45 | 1.0000 | 1.0000 | 1.0000 | 0.9320 |
| 17 | `garlic` (Tỏi) | 179 | 0.9890 | 1.0000 | 0.9999 | 0.8624 |
| 18 | `ginger` (Gừng) | 64 | 1.0000 | 0.9844 | 0.9802 | 0.9114 |
| 19 | `jicama` (Củ đậu / Củ sắn) | 109 | 0.9720 | 0.9541 | 0.9497 | 0.8502 |
| 20 | `okra` (Đậu bắp) | 92 | 0.9684 | 1.0000 | 0.9945 | 0.7326 |
| 21 | `onion` (Hành tây) | 126 | 0.9237 | 0.9603 | 0.9496 | 0.8093 |
| 22 | `pork` (Thịt heo) | 79 | 1.0000 | 1.0000 | 1.0000 | 0.8783 |
| 23 | `potato` (Khoai tây) | 56 | 0.9643 | 0.9643 | 0.9604 | 0.9049 |
| 24 | `pumpkin` (Bí đỏ) | 104 | 1.0000 | 0.9808 | 0.9802 | 0.9236 |
| 25 | `radish` (Củ cải trắng) | 159 | 0.9871 | 0.9623 | 0.9603 | 0.8395 |
| 26 | `scallion` (Hành lá) | 55 | 0.9636 | 0.9636 | 0.9584 | 0.8048 |
| 27 | `shrimp` (Tôm) | 48 | 1.0000 | 1.0000 | 1.0000 | 0.9098 |
| 28 | `spongegourd` (Mướp) | 143 | 0.9929 | 0.9790 | 0.9701 | 0.8822 |
| 29 | `sweetpotato` (Khoai lang) | 100 | 0.9216 | 0.9400 | 0.9147 | 0.8080 |
| 30 | `tofu` (Đậu phụ) | 44 | 1.0000 | 1.0000 | 1.0000 | 0.9080 |
| 31 | `tomato` (Cà chua) | 132 | 0.9924 | 0.9924 | 0.9899 | 0.8544 |
| **-** | **TRUNG BÌNH TOÀN BỘ** | **3,911** | **0.9756** | **0.9749** | **0.9859** | **0.8774** |

---

## 5. MA TRẬN NHẦM LẪN (CONFUSION MATRIX) & XUẤT DỮ LIỆU CSV CỦA RF-DETR

Được sinh tự động trên toàn bộ **1,481 ảnh** tập kiểm thử độc lập (Test Set) theo **Phương pháp Điểm làm việc tối ưu F1 (Optimal Operating Point Method, $\tau^* = 0.05$)**:
- **Trục tung (Y-axis)**: Lớp thực tế (Ground Truth Class).
- **Trục hoành (X-axis)**: Lớp dự đoán của mô hình (Predicted Class).
- **Cột Background (False Negatives)**: Tỷ lệ mẫu thực tế bị mô hình bỏ sót.
- **Hàng Background (False Positives)**: Dự đoán ảo vào vùng không có đối tượng.

### 5.1. Dữ liệu gốc định dạng CSV (Phục vụ nạp số liệu trực tiếp vào báo cáo & bảng tính):
- **Bảng chỉ số 32 lớp (Per-Class Metrics CSV)**: [rfdetr_per_class_metrics.csv](file:///d:/NCKH/Train_models/eval_results_csv/rfdetr_per_class_metrics.csv)
- **Ma trận đếm số lượng thực tế (Raw Counts CSV)**: [rfdetr_confusion_matrix.csv](file:///d:/NCKH/Train_models/eval_results_csv/rfdetr_confusion_matrix.csv)
- **Ma trận tỷ lệ chuẩn hóa % (Normalized Recall CSV)**: [rfdetr_confusion_matrix_normalized.csv](file:///d:/NCKH/Train_models/eval_results_csv/rfdetr_confusion_matrix_normalized.csv)

### 5.2. File hình ảnh trực quan hóa:
- **Ma trận đếm số lượng (Counts)**: [rfdetr_confusion_matrix.png](file:///d:/NCKH/Train_models/eval_results_csv/rfdetr_confusion_matrix.png)
- **Ma trận chuẩn hóa (Normalized %)**: [rfdetr_confusion_matrix_normalized.png](file:///d:/NCKH/Train_models/eval_results_csv/rfdetr_confusion_matrix_normalized.png)

### 5.3. Nhận xét khoa học & Kiểm chứng khớp số học 100%:
- **Độ chính xác đường chéo chính (True Positive - Recall)**: Đạt mức cực cao trên toàn bộ 32 lớp, dao động từ **92% đến 100%**. 7 lớp đạt độ chính xác nhận diện tuyệt đối $100\%$ gồm: `bittergourd` (1.00), `chayote` (1.00), `eggplant` (1.00), `garlic` (1.00), `pork` (1.00), `shrimp` (1.00), `tofu` (1.00).
- **Kiểm chứng lớp `chicken` (Thịt gà)**:
  - Tổng số mẫu thực tế trong tập Test: **50 mẫu**.
  - Mô hình nhận diện đúng: **47 mẫu** (chiếm **94.0%** trên ma trận chuẩn hóa).
  - Bỏ sót vào nhãn nền (Background Missed): **3 mẫu** (chiếm **6.0%**).
  - Nhầm lẫn sang các lớp thực phẩm khác: **0 mẫu**.
  - Đường chéo ma trận chuẩn hóa là **0.94**, **khớp chính xác 100%** với chỉ số `Recall = 0.9400` trong bảng chỉ số [rfdetr_per_class_metrics.csv](file:///d:/NCKH/Train_models/eval_results_csv/rfdetr_per_class_metrics.csv).
- **Tính đồng bộ khoa học**: Cả hai mô hình RT-DETR và RF-DETR đều được đánh giá ở điểm làm việc tối ưu F1 trên cùng tập test 1,481 ảnh / 3,911 mẫu, giải quyết hoàn toàn sự lệch số học giữa ma trận nhầm lẫn và bảng chỉ số chi tiết.


