# BÁO CÁO KHOA HỌC HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH MỚI: RF-DETR (LẦN 3 - RETRAINED)

**Dự án:** Nghiên cứu Khoa học - Nhận diện Thành phần Món ăn & Dinh dưỡng Việt Nam (NCKH CICT 2025-2026)  
**Tập dữ liệu:** `completed-project-5` (14,810 ảnh, 38,010 nhãn, 32 lớp món ăn & nguyên liệu)  
**Tập kiểm thử Test Split:** 1,481 ảnh độc lập (3,911 instances)  
**Môi trường thực thi:** Cloud Modal GPU NVIDIA A100-SXM4 (40GB VRAM, Persistent Volume Storage)  
**Kiến trúc mô hình:** `RF-DETR-Medium` (DINOv2 Foundation Backbone + Transformer Decoder)  
**Tổng số Epochs:** 27 Epochs (Early Stopping tại epoch 27, kích hoạt bởi cơ chế kiên nhẫn `patience = 10`)  
**Ngày hoàn thành:** 08/09/2026  

---

## 1. THÔNG SỐ CẤU HÌNH HUẤN LUYỆN CHI TIẾT (TRAINING CONFIGURATION)

| Thành phần (Component) | Thông số Thiết lập (Configuration) | Ý nghĩa Kỹ thuật |
| :--- | :--- | :--- |
| **Mô hình (Model Architecture)** | **RF-DETR Medium** (`RFDETRMedium`) | Backbone DINOv2 + Multi-scale Deformable Attention |
| **Phần cứng GPU** | **NVIDIA A100-SXM4 (40GB VRAM)** | Băng thông 1,935 GB/s, bfloat16 Automatic Mixed Precision |
| **Kích thước ảnh (Resolution)** | **$576 \times 576$ pixels** (`square_resize_div_64`) | Chuẩn hóa tỉ lệ lưới 36x36 positional encoding |
| **Multi-Scale Training** | **Được kích hoạt (`multi_scale: true, expanded_scales: true`)** | Học nhận diện đối tượng ở nhiều dải kích thước khác nhau |
| **Kích thước Batch (Batch Size)** | **8 ảnh / GPU** $\times$ **Gradient Accumulation: 2** | Kích thước batch hiệu dụng tương đương **16 ảnh/batch** |
| **Tốc độ học (Learning Rate)** | **$\text{lr} = 1.0 \times 10^{-4}$**, $\text{lr}_{\text{encoder}} = 1.5 \times 10^{-4}$ | Tốc độ học riêng biệt tối ưu cho Encoder và Decoder |
| **Bộ tối ưu (Optimizer)** | **AdamW** (`weight_decay: 0.0001, clip_max_norm: 0.1`) | Chống phân kỳ gradient trong kiến trúc Transformer |
| **Cơ chế EMA (Exponential Moving Average)**| **Kích hoạt (`ema_decay: 0.993, ema_tau: 100`)** | Trọng số EMA `best.pth` mượt mà, hạn chế nhiễu cục bộ |
| **Hệ thống dừng sớm (Early Stopping)** | **`patience = 10` epochs, `min_delta = 0.001`** | Ngắt huấn luyện tự động khi Validation không cải thiện |

---

## 2. KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ TRÊN TẬP TEST ĐỘC LẬP (TEST SPLIT OVERALL BENCHMARK)

Đo lường độc lập trên toàn bộ **1,481 ảnh (3,911 instances)** của tập Test split v5:

| Chỉ số Đánh giá (Evaluation Metric) | Kết Quả Đạt Được | Tỷ lệ Phần trăm (%) | Ý nghĩa Khoa học & Thực tiễn |
| :--- | :---: | :---: | :--- |
| **mAP@50 (IoU = 0.50)** | **0.9881** | **98.81%** | 🏆 **Kỷ lục cao nhất từ trước đến nay** (Vượt cả RT-DETR 98.56%) |
| **mAP@75 (IoU = 0.75)** | **0.9574** | **95.74%** | Khả năng bao khung bounding box sắc nét và chuẩn xác vượt bậc |
| **mAP@50-95 (COCO Standard)** | **0.8775** | **87.75%** | Chỉ số tổng hợp toàn diện theo chuẩn COCO Benchmark |
| **mAR (Mean Average Recall)** | **0.9198** | **91.98%** | Tỷ lệ thu hồi đối tượng trung bình trên mọi ngưỡng IoU |
| **Precision (Độ chính xác)** | **0.9739** | **97.39%** | Tỷ lệ nhận diện chuẩn xác cao, giảm tối đa báo động giả |
| **Recall (Độ nhạy / Thu hồi)** | **0.9751** | **97.51%** | Khả năng bắt trúng gần như toàn bộ thành phần thực phẩm |
| **F1-Score** | **0.9744** | **97.44%** | Điểm cân bằng hoàn hảo giữa độ chính xác và độ nhạy |
| **Test Loss** | **3.3956** | — | Tổng tổn thất trên tập kiểm thử test |

---

## 3. BẢNG ĐÁNH GIÁ CHI TIẾT TỪNG LỚP TRÊN TẬP TEST (PER-CLASS ACCURACY)

*Dữ liệu đo đạc thực tế chính xác trên toàn bộ 32 lớp thực phẩm tại tập Test split v5 (1,481 ảnh, 3,911 instances):*

| Class ID | Tên Lớp Thực Phẩm (Class Name) | Số Lượng (Instances) | Precision (P) | Recall (R) | F1-Score | mAR | mAP@50-95 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **beef** (Thịt bò) | 331 | 98.19% | 98.19% | 98.19% | 94.47% | 89.38% |
| **1** | **bellpepper** (Ớt chuông) | 83 | 95.29% | 97.59% | 96.43% | 95.54% | 93.29% |
| **2** | **bittergourd** (Khổ qua / Mướp đắng) | 50 | 100.00% | 100.00% | 100.00% | 97.00% | 95.37% |
| **3** | **bottlegourd** (Quả bầu) | 140 | 94.24% | 93.57% | 93.91% | 91.86% | 87.28% |
| **4** | **broccoli** (Súp lơ xanh) | 68 | 97.01% | 95.59% | 96.30% | 97.21% | 94.18% |
| **5** | **cabbage** (Bắp cải) | 79 | 95.00% | 96.20% | 95.60% | 92.91% | 89.73% |
| **6** | **carrot** (Cà rốt) | 74 | 96.05% | 98.65% | 97.33% | 88.92% | **84.88%** |
| **7** | **cauliflower** (Súp lơ trắng) | 109 | 93.97% | 100.00% | 96.89% | 96.24% | 92.77% |
| **8** | **chayote** (Quả su su) | 57 | 100.00% | 100.00% | 100.00% | 99.12% | 98.71% |
| **9** | **chicken** (Thịt gà) | 50 | 97.92% | 94.00% | 95.92% | 85.20% | **72.44%** |
| **10** | **chickenegg** (Trứng gà) | 223 | 98.67% | 99.55% | 99.11% | 92.51% | 90.52% |
| **11** | **chickenleg** (Đùi gà) | 531 | 99.05% | 98.49% | 98.77% | 84.58% | **79.98%** |
| **12** | **chickenwin** (Cánh gà) | 151 | 100.00% | 99.34% | 99.67% | 87.09% | **83.34%** |
| **13** | **corn** (Bắp ngô) | 134 | 96.95% | 94.78% | 95.85% | 87.84% | **80.75%** |
| **14** | **cucumber** (Dưa chuột) | 197 | 90.36% | 90.36% | 90.36% | 93.30% | 87.36% |
| **15** | **duckegg** (Trứng vịt) | 99 | 100.00% | 96.97% | 98.46% | 94.04% | 89.31% |
| **16** | **eggplant** (Cà tím) | 45 | 100.00% | 100.00% | 100.00% | 93.11% | 90.30% |
| **17** | **garlic** (Tỏi) | 179 | 96.70% | 98.32% | 97.51% | 89.44% | 85.33% |
| **18** | **ginger** (Gừng) | 64 | 100.00% | 100.00% | 100.00% | 94.53% | 92.05% |
| **19** | **jicama** (Củ đậu) | 109 | 97.20% | 95.41% | 96.30% | 90.64% | 86.09% |
| **20** | **okra** (Đậu bắp) | 92 | 96.81% | 98.91% | 97.85% | 81.52% | **73.22%** |
| **21** | **onion** (Hành tây) | 126 | 93.13% | 96.83% | 94.94% | 87.38% | **82.68%** |
| **22** | **pork** (Thịt heo) | 79 | 100.00% | 100.00% | 100.00% | 91.77% | 87.66% |
| **23** | **potato** (Khoai tây) | 56 | 100.00% | 98.21% | 99.10% | 96.79% | 93.56% |
| **24** | **pumpkin** (Bí đỏ) | 104 | 98.11% | 100.00% | 99.05% | 96.73% | 94.39% |
| **25** | **radish** (Củ cải trắng) | 159 | 98.73% | 98.11% | 98.42% | 90.19% | 86.67% |
| **26** | **scallion** (Hành lá) | 55 | 96.30% | 94.55% | 95.41% | 90.73% | 85.03% |
| **27** | **shrimp** (Tôm) | 48 | 100.00% | 100.00% | 100.00% | 94.17% | 91.08% |
| **28** | **spongegourd** (Mướp hương) | 143 | 100.00% | 98.60% | 99.30% | 93.50% | 90.11% |
| **29** | **sweetpotato** (Khoai lang) | 100 | 87.50% | 91.00% | 89.22% | 90.80% | **83.52%** |
| **30** | **tofu** (Đậu phụ) | 44 | 100.00% | 100.00% | 100.00% | 94.55% | 90.44% |
| **31** | **tomato** (Cà chua) | 132 | 99.22% | 96.97% | 98.08% | 89.85% | 86.48% |
| **ALL** | **Trung bình toàn bộ 32 lớp** | **3,911** | **97.39%** | **97.51%** | **97.44%** | **91.98%** | **87.75%** |

*(Ghi chú: Các chỉ số mAP@50-95 in đậm là các lớp có mAP < 85.00%).*

---

## 4. BẢNG LOG CHI TIẾT 27 EPOCHS HUẤN LUYỆN (FULL 27-EPOCH TRAINING LOG)

*Dữ liệu trích xuất từ file log `rfdetr_epoch_logs.csv`:*

| Epoch | Train Loss | Class Loss | Box Loss | GIoU Loss | Epoch Time (s) | Val Precision | Val Recall | Val mAP@50 | Val mAP@50-95 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 5.66341 | 0.65636 | 0.03855 | 0.09099 | 830.54 s | 0.9396 | 0.9409 | 0.9689 | 0.8380 |
| 2 | 4.55693 | 0.42014 | 0.03725 | 0.08835 | 285.86 s | 0.9571 | 0.9637 | 0.9813 | 0.8556 |
| 3 | 4.28615 | 0.38331 | 0.03652 | 0.08693 | 284.08 s | 0.9668 | 0.9625 | 0.9835 | 0.8598 |
| 4 | 4.16223 | 0.36509 | 0.03660 | 0.08714 | 286.87 s | 0.9611 | 0.9680 | 0.9842 | 0.8636 |
| 5 | 3.90162 | 0.34318 | 0.03510 | 0.08368 | 292.14 s | 0.9727 | 0.9696 | 0.9861 | 0.8672 |
| 6 | 3.89488 | 0.33803 | 0.03463 | 0.08328 | 290.38 s | 0.9714 | 0.9667 | 0.9853 | 0.8684 |
| 7 | 3.86932 | 0.33218 | 0.03444 | 0.08270 | 291.66 s | 0.9672 | 0.9724 | 0.9868 | 0.8700 |
| 8 | 3.80955 | 0.32585 | 0.03404 | 0.08210 | 283.25 s | 0.9747 | 0.9670 | 0.9865 | 0.8689 |
| 9 | 3.74246 | 0.32157 | 0.03338 | 0.08080 | 286.27 s | 0.9762 | 0.9637 | 0.9866 | 0.8707 |
| 10 | 3.68399 | 0.31882 | 0.03310 | 0.08041 | 281.58 s | 0.9719 | 0.9709 | 0.9865 | 0.8708 |
| 11 | 3.65576 | 0.31297 | 0.03295 | 0.08010 | 274.92 s | 0.9719 | 0.9727 | 0.9870 | 0.8724 |
| 12 | 3.62967 | 0.30664 | 0.03265 | 0.07976 | 284.08 s | 0.9727 | 0.9700 | 0.9860 | 0.8713 |
| 13 | 3.58868 | 0.30579 | 0.03235 | 0.07910 | 281.53 s | 0.9756 | 0.9723 | 0.9871 | 0.8714 |
| 14 | 3.57292 | 0.30199 | 0.03207 | 0.07831 | 277.56 s | 0.9752 | 0.9722 | 0.9876 | 0.8723 |
| 15 | 3.52525 | 0.29862 | 0.03200 | 0.07795 | 277.90 s | 0.9761 | 0.9700 | 0.9868 | 0.8715 |
| 16 | 3.52332 | 0.29660 | 0.03158 | 0.07747 | 279.89 s | 0.9736 | 0.9746 | 0.9872 | 0.8728 |
| 17 | 3.51414 | 0.29390 | 0.03100 | 0.07681 | 280.63 s | 0.9735 | 0.9723 | 0.9875 | 0.8739 |
| **18** | **3.46517** | **0.29260** | **0.03124** | **0.07705** | **281.23 s** | **0.9733** | **0.9750** | **0.9870** | **0.8744 (Best Val)** |
| 19 | 3.44827 | 0.29138 | 0.03102 | 0.07651 | 276.48 s | 0.9740 | 0.9741 | 0.9862 | 0.8722 |
| 20 | 3.42968 | 0.28490 | 0.03000 | 0.07478 | 276.95 s | 0.9726 | 0.9726 | 0.9872 | 0.8731 |
| 21 | 3.36735 | 0.28214 | 0.03022 | 0.07546 | 279.97 s | 0.9744 | 0.9695 | 0.9858 | 0.8724 |
| 22 | 3.35737 | 0.28254 | 0.02989 | 0.07482 | 278.53 s | 0.9751 | 0.9696 | 0.9860 | 0.8726 |
| 23 | 3.31491 | 0.27832 | 0.02950 | 0.07398 | 275.30 s | 0.9730 | 0.9742 | 0.9857 | 0.8727 |
| 24 | 3.27467 | 0.27686 | 0.02964 | 0.07384 | 277.98 s | 0.9695 | 0.9763 | 0.9860 | 0.8732 |
| 25 | 3.25204 | 0.27606 | 0.02924 | 0.07319 | 279.86 s | 0.9758 | 0.9681 | 0.9852 | 0.8724 |
| 26 | 3.23338 | 0.27203 | 0.02907 | 0.07276 | 282.10 s | 0.9776 | 0.9712 | 0.9860 | 0.8738 |
| 27 | 3.22708 | 0.27346 | 0.02888 | 0.07257 | 283.35 s | 0.9761 | 0.9735 | 0.9861 | 0.8738 |

---

## 5. PHÂN TÍCH CHUYÊN SÂU & ĐÁNH GIÁ KỸ THUẬT (TECHNICAL ANALYSIS)

### 5.1. Tốc độ Hội tụ và Đỉnh Cao Hiệu Năng
1. **Hội tụ siêu tốc nhờ DINOv2 & Multi-scale Attention**:
   - Ngay từ **Epoch 2**, mô hình đã đạt `val_mAP50 = 0.9813` và `val_mAP50_95 = 0.8556`.
   - Hàm mất mát `Train Loss` giảm đều đặn từ **5.66341** xuống **3.22708**, trong đó `Class Loss` giảm hơn một nửa từ **0.65636** xuống **0.27203**, khẳng định khả năng phân biệt 32 lớp nguyên liệu rất vững chắc.
2. **Điểm tối ưu Validation (Best Checkpoint tại Epoch 18)**:
   - Điểm kiểm định cao nhất đạt được tại **Epoch 18** với `val_mAP50_95 = 0.8744`, `val_mAP50 = 0.9870`, `Val Precision = 0.9733`, `Val Recall = 0.9750`.
   - So với lần huấn luyện thứ 2 (`val_mAP50_95 = 0.8670` tại Epoch 30), lần 3 đạt mốc kiểm định Validation cao hơn **+0.74%** và sớm hơn 12 epochs.
3. **Cơ chế Early Stopping hiệu quả**:
   - Từ Epoch 19 đến 27, chỉ số validation duy trì mức cao ổn định ($0.8722 - 0.8738$) và không vượt qua ngưỡng đỉnh $0.8744$.
   - Cơ chế Early Stopping (`patience = 10`) tự động dừng huấn luyện tại Epoch 27, giúp tiết kiệm gần 50% thời gian và ngân sách GPU mà vẫn lưu giữ được trọng số tối ưu nhất (`best.pth`).

### 5.2. Đột Phá Ở Các Lớp Yếu Điển Hình
- **`chicken` (Thịt gà)**: Tăng vọt từ **`68.33%`** (ở lần 2) lên **`72.44%`** (**+4.11%**). Nhờ multi-scale training 576-736, mô hình nhận diện tốt hơn các miếng thịt gà chặt khúc kích thước biến thiên.
- **`chayote` (Quả su su)**: Đạt đỉnh tuyệt đối **`98.71%`** (Precision 100%, Recall 100%).
- **`bittergourd` (Khổ qua)**: **`95.37%`** (Precision 100%, Recall 100%).
- **`potato` (Khoai tây)**: **`93.56%`** (Precision 100%, Recall 98.21%).
- **`garlic` (Tỏi)**: Đạt **`85.33%`** và **`scallion` (Hành lá)**: Đạt **`85.03%`** (vượt mốc 85%).

### 5.3. Tối Ưu Hóa Tốc Độ Thực Thi Trên Mỗi Epoch
- Ở lần thứ 2, mỗi epoch mất trung bình **$750\text{s} - 820\text{s}$**.
- Ở lần thứ 3 (mới), mỗi epoch chỉ mất **$275\text{s} - 290\text{s}$** (ngoại trừ epoch 1 khởi động).
- **Tốc độ thực thi nhanh gấp 2.6 lần**, giúp toàn bộ 27 epochs hoàn tất chỉ trong khoảng **2.3 giờ** tính toán trên Modal GPU A100.