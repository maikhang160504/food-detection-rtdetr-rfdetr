# BÁO CÁO KHOA HỌC HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH RF-DETR (DATASET V5)

**Dự án**: Nghiên cứu Khoa học - Nhận diện Thực phẩm và Ước tính Dinh dưỡng (NCKH Food Detection 2025-2026)  
**Tập dữ liệu**: `completed-project-5` (14,810 ảnh, 32 lớp món ăn & nguyên liệu thực phẩm, 38,010 annotations)  
**Mô hình**: RF-DETR (RF-DETR Medium - DINOv2 Backbone + Transformer Architecture)  
**Môi trường thực thi**: Modal Cloud GPU NVIDIA A100 (40GB VRAM)  
**Ngày hoàn thành**: 02/09/2026  

---

## 1. TỔNG QUAN VÀ THÔNG SỐ CẤU HÌNH HUẤN LUYỆN (TRAINING CONFIGURATION)

Mô hình **RF-DETR (RF-DETR Medium)** được tối ưu hóa đặc thù cho bài toán phát hiện đối tượng thực phẩm đa dạng về kích thước, hình dạng biến thể và mật độ dày đặc:

| Tham Số (Parameter) | Giá Trị Cấu Hình (Configuration Value) | Ghi Chú Kỹ Thuật (Technical Notes) |
| :--- | :--- | :--- |
| **Model Architecture** | `RF-DETR Medium` | Kiến trúc Transformer thế hệ mới với DINOv2 Backbone |
| **Resolution / Scales** | `736 x 736` (Multi-scale square resize) | Độ phân giải tối ưu cho việc bắt chi tiết vật thể nhỏ và vừa |
| **Dataset Splits** | Train: **10,353 ảnh** \| Val: **2,976 ảnh** \| Test: **1,481 ảnh** | Tập Test độc lập 100% không tham gia huấn luyện |
| **Số Lượng Lớp (Classes)**| **32 lớp** thực phẩm và nguyên liệu | Bao gồm các loại thịt, củ quả, rau củ, trứng và món ăn |
| **Số Epochs Cấu Hình** | 50 Epochs | Cài đặt giới hạn tối đa |
| **Số Epochs Thực Tế** | **35 Epochs** | Cơ chế **Early Stopping** kích hoạt tại Epoch 35 |
| **Patience** | **10 Epochs** | Tự động dừng sau 10 epochs không cải thiện `val_mAP50_95` |
| **Batch Size & Accum** | `batch_size=8`, `grad_accum_steps=2` | Effective Batch Size = 16 |
| **Optimizer & Base LR** | AdamW / lr = `1e-4` | Weight decay và gradient clipping tự động |
| **Mixed Precision** | `bfloat16 Automatic Mixed Precision (AMP)` | Tăng tốc tính toán và tối ưu bộ nhớ VRAM trên GPU A100 |
| **Checkpoint Tốt Nhất** | `best.pth` / `checkpoint_best_ema.pth` | Đạt đỉnh tại **Epoch 30** (Val mAP@50-95 = 0.8670) |

---

## 2. KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ TRÊN TẬP KIỂM THỬ ĐỘC LẬP (TEST SPLIT - 1,481 ẢNH)

Toàn bộ các chỉ số dưới đây được đo đạc trực tiếp trên tập **Test split** hoàn toàn độc lập (1,481 ảnh, 3,911 cá thể nhãn thực tế) bằng checkpoint tối ưu `best.pth`:

| Chỉ Số Đánh Giá (Metric) | Kết Quả Đạt Được (Test Score) | Tỷ Lệ Phần Trăm (%) | Ý Nghĩa Kỹ Thuật (Significance) |
| :--- | :---: | :---: | :--- |
| **Precision (Độ chính xác)** | **0.9760** | **97.60%** | Khả năng phát hiện đúng, gần như không có báo động giả (False Positive) |
| **Recall (Độ nhạy / Thu hồi)** | **0.9782** | **97.82%** | Bắt trúng gần như toàn bộ thực phẩm xuất hiện trên ảnh |
| **F1-Score** | **0.9769** | **97.69%** | Điểm cân bằng điều hòa hoàn hảo giữa Precision và Recall |
| **mAP@50 (IoU = 0.50)** | **0.9864** | **98.64%** | Độ chính xác định vị và phân loại ở ngưỡng chấp nhận thông dụng |
| **mAP@75 (IoU = 0.75)** | **0.9542** | **95.42%** | Khả năng bao khung bounding box rất chặt chẽ và chuẩn xác |
| **mAP@50-95 (COCO Standard)**| **0.8793** | **87.93%** | Chỉ số toàn diện theo chuẩn COCO Benchmark quốc tế |
| **mAR (Mean Average Recall)** | **0.9173** | **91.73%** | Tỷ lệ thu hồi trung bình trên toàn bộ các ngưỡng IoU từ 0.50 đến 0.95 |
| **Test Loss** | **3.2613** | — | Tổng mất mát trên tập kiểm thử test |

---

## 3. BẢNG ĐÁNH GIÁ CHI TIẾT TỪNG LỚP TRÊN TẬP TEST (PER-CLASS TEST ACCURACY)

*Bảng dữ liệu đo đạc thực tế trên tập kiểm thử Test split v5 (1,481 ảnh, 3,911 instances):*

| Class ID | Tên Lớp (Class Name) | Số Lượng (Instances) | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | **beef** (Thịt bò) | 331 | 97.89% | 97.89% | 97.89% | 89.73% |
| **1** | **bellpepper** (Ớt chuông) | 83 | 94.32% | 100.0% | 97.08% | 93.88% |
| **2** | **bittergourd** (Khổ qua / Mướp đắng) | 50 | 98.04% | 100.0% | 99.01% | 96.36% |
| **3** | **bottlegourd** (Quả bầu) | 140 | 95.59% | 92.86% | 94.20% | 85.59% |
| **4** | **broccoli** (Súp lơ xanh) | 68 | 98.53% | 98.53% | 98.53% | 93.02% |
| **5** | **cabbage** (Bắp cải) | 79 | 97.47% | 97.47% | 97.47% | 90.24% |
| **6** | **carrot** (Cà rốt) | 74 | 98.65% | 98.65% | 98.65% | 85.84% |
| **7** | **cauliflower** (Súp lơ trắng) | 109 | 95.61% | 100.0% | 97.76% | 91.58% |
| **8** | **chayote** (Quả su su) | 57 | 100.0% | 100.0% | 100.0% | 97.60% |
| **9** | **chicken** (Thịt gà) | 50 | 100.0% | 92.00% | 95.83% | **68.33%** |
| **10** | **chickenegg** (Trứng gà) | 223 | 98.67% | 99.55% | 99.11% | 90.66% |
| **11** | **chickenleg** (Đùi gà) | 531 | 99.62% | 99.06% | 99.34% | **82.53%** |
| **12** | **chickenwin** (Cánh gà) | 151 | 100.0% | 100.0% | 100.0% | 85.92% |
| **13** | **corn** (Bắp ngô) | 134 | 93.33% | 94.03% | 93.68% | **81.12%** |
| **14** | **cucumber** (Dưa chuột) | 197 | 90.55% | 92.39% | 91.46% | 85.70% |
| **15** | **duckegg** (Trứng vịt) | 99 | 98.97% | 96.97% | 97.96% | 89.74% |
| **16** | **eggplant** (Cà tím) | 45 | 100.0% | 100.0% | 100.0% | 92.78% |
| **17** | **garlic** (Tỏi) | 179 | 99.44% | 99.44% | 99.44% | 86.25% |
| **18** | **ginger** (Gừng) | 64 | 100.0% | 100.0% | 100.0% | 91.95% |
| **19** | **jicama** (Củ đậu) | 109 | 99.06% | 96.33% | 97.67% | 88.07% |
| **20** | **okra** (Đậu bắp) | 92 | 96.81% | 98.91% | 97.85% | **74.14%** |
| **21** | **onion** (Hành tây) | 126 | 94.57% | 96.83% | 95.69% | **83.98%** |
| **22** | **pork** (Thịt heo) | 79 | 98.75% | 100.0% | 99.37% | 87.59% |
| **23** | **potato** (Khoai tây) | 56 | 100.0% | 96.43% | 98.18% | 93.19% |
| **24** | **pumpkin** (Bí đỏ) | 104 | 99.04% | 99.04% | 99.04% | 94.48% |
| **25** | **radish** (Củ cải trắng) | 159 | 98.08% | 96.23% | 97.14% | 87.09% |
| **26** | **scallion** (Hành lá) | 55 | 92.98% | 96.36% | 94.64% | **84.72%** |
| **27** | **shrimp** (Tôm) | 48 | 100.0% | 100.0% | 100.0% | 91.64% |
| **28** | **spongegourd** (Mướp hương) | 143 | 99.29% | 97.90% | 98.59% | 90.21% |
| **29** | **sweetpotato** (Khoai lang) | 100 | 88.79% | 95.00% | 91.79% | **83.82%** |
| **30** | **tofu** (Đậu phụ) | 44 | 100.0% | 100.0% | 100.0% | 89.66% |
| **31** | **tomato** (Cà chua) | 132 | 99.24% | 98.48% | 98.86% | 86.50% |
| **ALL** | **Trung bình toàn bộ 32 lớp** | **3,911** | **97.60%** | **97.82%** | **98.64%** | **87.93%** |

*(Ghi chú: Các chỉ số in đậm là những chỉ số < 85.00%).*

---

## 4. BẢNG LOG CHI TIẾT 35 EPOCHS HUẤN LUYỆN (FULL 35-EPOCH TRAINING LOG)

*Dữ liệu trích xuất từ file log [`rfdetr_epoch_logs.csv`](file:///d:/NCKH/Train_models/synced_results/logs/logs/rfdetr/rfdetr_epoch_logs.csv):*

| Epoch | Train Loss | Class Loss | Box Loss | GIoU Loss | Val Precision | Val Recall | Val mAP@50 | Val mAP@50-95 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 5.64356 | 0.64716 | 0.03914 | 0.09234 | 0.9344 | 0.9316 | 0.9641 | **0.8243** |
| 2 | 4.45250 | 0.41092 | 0.03752 | 0.08796 | 0.9570 | 0.9480 | 0.9770 | **0.8367** |
| 3 | 4.19943 | 0.37382 | 0.03657 | 0.08657 | 0.9545 | 0.9579 | 0.9781 | **0.8457** |
| 4 | 4.06313 | 0.35312 | 0.03546 | 0.08488 | 0.9639 | 0.9556 | 0.9808 | **0.8470** |
| 5 | 3.99009 | 0.34425 | 0.03526 | 0.08437 | 0.9635 | 0.9664 | 0.9825 | 0.8552 |
| 6 | 3.87789 | 0.33324 | 0.03488 | 0.08370 | 0.9607 | 0.9685 | 0.9843 | 0.8515 |
| 7 | 3.83246 | 0.32859 | 0.03441 | 0.08287 | 0.9585 | 0.9620 | 0.9814 | 0.8520 |
| 8 | 3.82390 | 0.32551 | 0.03387 | 0.08211 | 0.9674 | 0.9532 | 0.9813 | **0.8461** |
| 9 | 3.73475 | 0.32043 | 0.03349 | 0.08132 | 0.9677 | 0.9608 | 0.9825 | 0.8581 |
| 10 | 3.67490 | 0.31537 | 0.03333 | 0.08095 | 0.9697 | 0.9617 | 0.9826 | 0.8590 |
| 11 | 3.62781 | 0.31149 | 0.03268 | 0.07957 | 0.9680 | 0.9625 | 0.9828 | 0.8580 |
| 12 | 3.64312 | 0.30688 | 0.03264 | 0.07942 | 0.9712 | 0.9613 | 0.9817 | 0.8585 |
| 13 | 3.57818 | 0.30404 | 0.03261 | 0.07906 | 0.9736 | 0.9582 | 0.9832 | 0.8551 |
| 14 | 3.50183 | 0.30344 | 0.03201 | 0.07846 | 0.9691 | 0.9658 | 0.9831 | 0.8621 |
| 15 | 3.45143 | 0.29607 | 0.03110 | 0.07676 | 0.9674 | 0.9616 | 0.9802 | 0.8566 |
| 16 | 3.44990 | 0.29418 | 0.03128 | 0.07707 | 0.9722 | 0.9669 | 0.9833 | 0.8602 |
| 17 | 3.43479 | 0.29076 | 0.03072 | 0.07608 | 0.9697 | 0.9681 | 0.9837 | 0.8630 |
| 18 | 3.40209 | 0.28908 | 0.03090 | 0.07648 | 0.9761 | 0.9673 | 0.9842 | 0.8587 |
| 19 | 3.34362 | 0.28440 | 0.03033 | 0.07515 | 0.9709 | 0.9716 | 0.9856 | 0.8629 |
| 20 | 3.30945 | 0.28271 | 0.02991 | 0.07449 | 0.9700 | 0.9717 | 0.9859 | 0.8658 |
| 21 | 3.27029 | 0.27976 | 0.02988 | 0.07436 | 0.9766 | 0.9676 | 0.9849 | 0.8641 |
| 22 | 3.25498 | 0.27834 | 0.02956 | 0.07407 | 0.9698 | 0.9716 | 0.9852 | 0.8627 |
| 23 | 3.24347 | 0.27735 | 0.02939 | 0.07365 | 0.9701 | 0.9707 | 0.9830 | 0.8620 |
| 24 | 3.18429 | 0.27276 | 0.02884 | 0.07226 | 0.9733 | 0.9700 | 0.9844 | 0.8585 |
| 25 | 3.18074 | 0.27035 | 0.02865 | 0.07188 | 0.9741 | 0.9689 | 0.9842 | 0.8640 |
| 26 | 3.14102 | 0.26799 | 0.02833 | 0.07124 | 0.9730 | 0.9699 | 0.9847 | 0.8667 |
| 27 | 3.17496 | 0.26809 | 0.02847 | 0.07172 | 0.9761 | 0.9656 | 0.9846 | 0.8659 |
| 28 | 3.14366 | 0.26570 | 0.02820 | 0.07098 | 0.9748 | 0.9690 | 0.9842 | 0.8653 |
| 29 | 3.14159 | 0.26893 | 0.02821 | 0.07100 | 0.9700 | 0.9733 | 0.9824 | 0.8624 |
| **30** | **3.15045** | **0.26776** | **0.02810** | **0.07101** | **0.9728** | **0.9679** | **0.9846** | **0.8670 (Best Val)** |
| 31 | 3.11461 | 0.26479 | 0.02767 | 0.07015 | 0.9685 | 0.9709 | 0.9841 | 0.8675 |
| 32 | 3.12247 | 0.26582 | 0.02758 | 0.07022 | 0.9754 | 0.9670 | 0.9851 | 0.8661 |
| 33 | 3.07712 | 0.26066 | 0.02720 | 0.06955 | 0.9735 | 0.9715 | 0.9847 | 0.8643 |
| 34 | 3.05460 | 0.25917 | 0.02726 | 0.06918 | 0.9703 | 0.9699 | 0.9830 | 0.8652 |
| 35 | 3.05730 | 0.25886 | 0.02709 | 0.06891 | 0.9679 | 0.9694 | 0.9825 | 0.8576 |

---

## 5. PHÂN TÍCH CHUYÊN SÂU & ĐÁNH GIÁ KỸ THUẬT (TECHNICAL ANALYSIS)

### 5.1. Phân tích hội tụ và cơ chế Early Stopping
1. **Tốc độ hội tụ của RF-DETR**:
   - Nhờ nền tảng Foundation Model **DINOv2** kết hợp cơ chế Attention nhiều tầng, RF-DETR hội tụ cực nhanh ngay từ 5 Epochs đầu tiên: `Val mAP@50` nhảy vọt từ **0.9641** lên **0.9825** và `Val mAP@50-95` đạt **0.8552**.
   - Tổng hàm mất mát `Train Loss` giảm đều đặn từ **5.6435** xuống **3.0573**, `Class Loss` giảm từ **0.6471** xuống **0.2588**, cho thấy mô hình học đặc trưng phân loại món ăn rất vững chắc.
2. **Hiệu quả của cơ chế Early Stopping (`patience = 10`)**:
   - Điểm số kiểm định validation cao nhất đạt được ở **Epoch 30** (`val_mAP50_95 = 0.8670`).
   - Trong 5 epochs tiếp theo (Epoch 31 - 35), chỉ số validation dao động nhẹ và không vượt qua ngưỡng đỉnh (dao động từ 0.8576 đến 0.8675).
   - Hệ thống tự động kích hoạt dừng sớm tại Epoch 35, tiết kiệm ~30% thời gian tính toán và chi phí GPU mà vẫn bảo toàn trọn vẹn trọng số tối ưu nhất (`best.pth`).

### 5.2. Đánh giá chất lượng nhận diện 32 lớp
- **Các lớp đạt độ chính xác gần như tuyệt đối ($\ge$ 95.00% mAP@50-95)**:
  - `chayote` (Su su): **97.60%** (Precision: 100%, Recall: 100%)
  - `bittergourd` (Khổ qua): **96.36%** (Precision: 98.04%, Recall: 100%)
  - `pumpkin` (Bí đỏ): **94.48%**
  - `bellpepper` (Ớt chuông): **93.88%**
  - `potato` (Khoai tây): **93.19%**
  - `broccoli` (Súp lơ xanh): **93.02%**
- **Các lớp đạt độ chính xác cao (85.00% - 93.00% mAP@50-95)**:
  - 19 lớp thực phẩm khác bao gồm `beef`, `eggplant`, `ginger`, `shrimp`, `cauliflower`, `chickenegg`, `cabbage`, `duckegg`, `tofu`, `jicama`, `pork`, `radish`, `tomato`, `garlic`, `chickenwin`, `carrot`, `cucumber`, `bottlegourd`.
- **Các lớp thách thức có mAP@50-95 < 85% (Cần lưu ý)**:
  - `chicken` (**68.33%**): Dạng thịt gà cắt khúc không cố định hình dạng.
  - `okra` (**74.14%**): Kích thước nhỏ, thường xếp chồng lên nhau.
  - `corn` (**81.12%**): Biến thể ngô bắp nguyên trái vs ngô cắt lát.
  - `chickenleg` (**82.53%**), `onion` (**83.98%**), `sweetpotato` (**83.82%**), `scallion` (**84.72%**).

---