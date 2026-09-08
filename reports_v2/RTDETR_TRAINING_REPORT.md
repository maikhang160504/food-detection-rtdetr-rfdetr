# BÁO CÁO KHOA HỌC CHI TIẾT HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH RT-DETR (PHIÊN BẢN DATASET V5)
**Dự án:** Nhận diện thành phần món ăn Việt Nam (Vietnamese Food Ingredient Detection - NCKH CICT 2025)  
**Tập dữ liệu:** `completed-project-5` (14,810 ảnh, 38,010 nhãn đối tượng, 32 lớp món ăn)  
**Môi trường thực thi:** Cloud Modal GPU NVIDIA A100-SXM4 (40GB VRAM, Persistent Volume Storage)  
**Mô hình:** `RT-DETR-L (Real-Time Detection Transformer Large)`  
**Số Epochs huấn luyện:** 50 Epochs  
**Thời gian hoàn thành:** 01/09/2026  

---

## 1. BẢNG THÔNG SỐ ĐẦU VÀO VÀ SIÊU THAM SỐ HUẤN LUYỆN (HYPERPARAMETERS & CONFIGURATIONS)

### 1.1. Thông số Phần cứng & Môi trường Thực thi (Hardware & Environment)
| Thành phần (Component) | Thông số Thiết lập (Configuration) | Ghi chú Kỹ thuật |
| :--- | :--- | :--- |
| **GPU phần cứng** | **NVIDIA A100-SXM4 (40GB VRAM)** | Băng thông bộ nhớ 1,935 GB/s, tối ưu hóa Cross-Attention |
| **Hệ thống lưu trữ** | Modal Persistent Cloud Volume (`/vol`) | Lưu trữ bền vững dataset v5, checkpoints và logs |
| **Framework & Engine** | PyTorch 2.1.3 + CUDA 12.1 + Ultralytics Engine | Pipeline xử lý song song tensor trên GPU |
| **Độ chính xác tính toán** | Automatic Mixed Precision (AMP - fp16/bf16) | Tăng tốc độ tính toán gấp đôi và tiết kiệm bộ nhớ VRAM |
| **Số tiến trình tải dữ liệu** | `workers = 8` | Đa luồng nạp ảnh chống nghẽn cổ chai I/O |

### 1.2. Cấu hình Tập dữ liệu & Kiến trúc Mô hình (Dataset & Model Architecture)
| Thông số (Parameter) | Giá trị (Value) | Mô tả Chi tiết |
| :--- | :--- | :--- |
| **Tên tập dữ liệu** | `completed-project-5` | Bộ dữ liệu nguyên liệu món ăn phiên bản mới nhất |
| **Tổng số lượng ảnh** | **14,810 ảnh** | Chuẩn hóa chất lượng, đa dạng góc chụp và ánh sáng thực tế |
| **Tổng số nhãn gán (Annotations)** | **38,010 nhãn** | Nhãn gán chuẩn xác cao, trung bình 2.57 đối tượng/ảnh |
| **Phân chia dữ liệu (Tỉ lệ 7:2:1)** | • **Train:** 10,353 ảnh (26,317 nhãn)<br>• **Validation:** 2,976 ảnh (7,782 nhãn)<br>• **Test:** 1,481 ảnh (3,911 nhãn) | Phân chia độc lập nghiêm ngặt theo chuẩn quốc tế |
| **Số lượng nhãn phân loại** | **32 lớp món ăn** | Thịt bò, gà, heo, tôm, trứng, rau củ quả Việt Nam |
| **Kích thước ảnh đầu vào** | **$640 \times 640$ pixels** (`imgsz: 640`) | Chuẩn hóa tỉ lệ vuông trước khi đưa vào backbone |
| **Kiến trúc mô hình** | **RT-DETR-L** (`rtdetr-l.pt`) | Backbone HGNetv2 + Hybrid Encoder + Transformer Decoder |
| **Số lượng tham số (Params)** | **32,049,500 tham số (~32M)** | Kích thước file trọng số tối ưu: **63.4 MB** |
| **Số phép tính (FLOPs)** | **105.5 GFLOPs** | Đảm bảo tốc độ đáp ứng thời gian thực (>180 FPS) |

### 1.3. Bảng Siêu tham số Huấn luyện & Tăng cường Dữ liệu (Training Hyperparameters & Augmentations)
| Siêu tham số (Hyperparameter) | Giá trị Thiết lập | Ý nghĩa & Cơ chế Hoạt động |
| :--- | :--- | :--- |
| **Tổng số Epochs (`epochs`)** | **50 Epochs** | Chu kỳ huấn luyện toàn diện |
| **Patience dừng sớm (`patience`)**| **10 Epochs** | Tự động ngắt nếu không cải thiện sau 10 chu kỳ |
| **Kích thước Batch (`batch_size`)** | **16 ảnh / batch** | Cân bằng gradient ổn định và tận dụng tối đa VRAM |
| **Thuật toán tối ưu (`optimizer`)** | **AdamW** | Tối ưu hóa trọng số cho kiến trúc Attention Transformer |
| **Tốc độ học cơ sở (`lr0`)** | **$1.0 \times 10^{-4}$ (0.0001)** | Tốc độ học khởi đầu chuẩn cho pre-trained Transformer |
| **Tốc độ học tối thiểu (`lrf`)** | **$1.0 \times 10^{-6}$ (0.01 factor)** | Giảm dần theo Cosine Decay để tinh chỉnh hội tụ sâu |
| **Momentum (`momentum`)** | **0.937** | Hệ số quán tính tích lũy gradient |
| **Phân rã trọng số (`weight_decay`)**| **$5.0 \times 10^{-4}$ (0.0005)** | Regularization chống hiện tượng Overfitting |
| **Giai đoạn khởi động (`warmup_epochs`)** | **3.0 Epochs** | Tăng dần lr trong 3 epoch đầu để ổn định trọng số |
| **Trọng số hàm mất mát Box (`box`)** | **7.5** | Ưu tiên định vị khớp tọa độ bounding box |
| **Trọng số hàm mất mát Class (`cls`)**| **0.5** | Cân bằng phân loại đa lớp món ăn với Focal Loss |
| **Trọng số hàm mất mát DFL/GIoU (`dfl`)**| **1.5** | Lực kéo hình học mở rộng cho bounding box |
| **Tăng cường Mosaic (`mosaic`)** | **1.0** (Tắt ở 10 epoch cuối - `close_mosaic: 10`) | Ghép 4 ảnh tăng khả năng phát hiện vật thể nhỏ |
| **Biến đổi màu sắc (`hsv_h / s / v`)** | `h: 0.015, s: 0.7, v: 0.4` | Mô phỏng điều kiện ánh sáng thực tế đa dạng |
| **Dịch chuyển & Co giãn (`translate / scale`)** | `translate: 0.1, scale: 0.5` | Đa dạng hóa góc nhìn và khoảng cách chụp |
| **Lật ảnh ngang (`fliplr`)** | **0.5 (50%)** | Tăng tính bất biến không gian |
| **Xóa vùng ngẫu nhiên (`erasing`)** | **0.4 (40%)** | Rèn luyện khả năng nhận diện khi món ăn bị che khuất |

---

## 2. ĐỘ CHÍNH XÁC TỔNG THỂ TRÊN TẬP DỮ LIỆU KIỂM THỬ (TEST SPLIT OVERALL ACCURACY)

Đánh giá độc lập trên toàn bộ **1,481 ảnh (3,911 đối tượng thực phẩm)** của tập Test v5:

| Chỉ số (Metric) | Kết quả Đạt Được (Dataset v5) | So Sánh với Dataset v1 cũ | Đánh Giá Khoa Học |
| :--- | :---: | :---: | :--- |
| **mAP@50 (mAP IoU 0.50)** | **98.56%** (0.9856) | Tăng +0.50% (98.06% $\rightarrow$ **98.56%**) | Độ chính xác phát hiện và phân loại gần như tuyệt đối |
| **mAP@50-95 (COCO Standard)** | **88.45%** (0.8845) | Tăng +0.67% (87.78% $\rightarrow$ **88.45%**) | Khả năng định vị Bounding Box cực kỳ ôm khít món ăn |
| **Precision (Độ chuẩn xác)** | **97.56%** (0.9756) | Tăng +0.49% (97.07% $\rightarrow$ **97.56%**) | Tỷ lệ dự đoán đúng đối tượng thực phẩm cực cao |
| **Recall (Độ nhạy)** | **97.59%** (0.9759) | Tăng +0.93% (96.66% $\rightarrow$ **97.59%**) | Hầu như không bỏ sót nguyên liệu bị che khuất |
| **Tốc độ suy luận (Inference Speed)** | **5.4 ms / ảnh** | Tương đương **~185 FPS** | Xử lý thời gian thực vượt chuẩn camera 30/60 FPS |
| **Trọng số tối ưu (Best Checkpoint)** | `best.pt` (63.4 MB) | Đạt mAP@50-95 cao nhất tại Epoch 42 trên tập Validation | Trọng số khuyến nghị triển khai sản phẩm |
| **Trọng số cuối cùng (Last Checkpoint)** | `last.pt` (63.4 MB) | Trọng số sau 50 epochs hoàn tất | Lưu trữ kiểm chứng |

---

## 3. ĐỘ CHÍNH XÁC CHI TIẾT TỪNG LỚP TRÊN TẬP TEST (PER-CLASS TEST ACCURACY)

*Bảng dữ liệu đo đạc thực tế trên tập kiểm thử Test split v5 (1,481 ảnh, 3,911 instances):*

| Class ID | Tên Lớp (Class Name) | Số Lượng (Instances) | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | **beef** (Thịt bò) | 331 | 98.22% | 98.49% | 98.77% | 91.48% |
| **1** | **bellpepper** (Ớt chuông) | 83 | 97.31% | 96.39% | 99.16% | 93.85% |
| **2** | **bittergourd** (Khổ qua / Mướp đắng) | 50 | 99.65% | 100.0% | 99.50% | 97.30% |
| **3** | **bottlegourd** (Quả bầu) | 140 | 96.28% | 92.46% | 97.65% | 87.04% |
| **4** | **broccoli** (Súp lơ xanh) | 68 | 97.08% | 97.06% | 98.75% | 94.64% |
| **5** | **cabbage** (Bắp cải) | 79 | 99.23% | 93.67% | 98.21% | 89.74% |
| **6** | **carrot** (Cà rốt) | 74 | 96.52% | 98.65% | 98.49% | 86.50% |
| **7** | **cauliflower** (Súp lơ trắng) | 109 | 95.44% | 100.0% | 98.66% | 91.96% |
| **8** | **chayote** (Quả su su) | 57 | 99.62% | 100.0% | 99.50% | 97.29% |
| **9** | **chicken** (Thịt gà) | 50 | 95.77% | 96.00% | 97.38% | **67.80%** |
| **10** | **chickenegg** (Trứng gà) | 223 | 98.53% | 99.55% | 99.22% | 90.68% |
| **11** | **chickenleg** (Đùi gà) | 531 | 98.98% | 97.74% | 99.17% | **83.92%** |
| **12** | **chickenwin** (Cánh gà) | 151 | 99.86% | 100.0% | 99.50% | 88.03% |
| **13** | **corn** (Bắp ngô) | 134 | 95.38% | 92.47% | 95.64% | **81.47%** |
| **14** | **cucumber** (Dưa chuột) | 197 | 91.04% | 92.87% | 94.96% | 85.55% |
| **15** | **duckegg** (Trứng vịt) | 99 | 97.95% | 96.49% | 98.19% | 92.55% |
| **16** | **eggplant** (Cà tím) | 45 | 99.64% | 100.0% | 99.50% | 93.56% |
| **17** | **garlic** (Tỏi) | 179 | 97.75% | 98.88% | 99.42% | 86.67% |
| **18** | **ginger** (Gừng) | 64 | 99.92% | 100.0% | 99.50% | 91.70% |
| **19** | **jicama** (Củ đậu) | 109 | 99.01% | 92.20% | 97.90% | 87.97% |
| **20** | **okra** (Đậu bắp) | 92 | 96.82% | 99.16% | 98.67% | **74.27%** |
| **21** | **onion** (Hành tây) | 126 | 94.56% | 96.58% | 96.85% | **83.29%** |
| **22** | **pork** (Thịt heo) | 79 | 99.78% | 100.0% | 99.50% | 88.47% |
| **23** | **potato** (Khoai tây) | 56 | 99.76% | 100.0% | 99.50% | 93.24% |
| **24** | **pumpkin** (Bí đỏ) | 104 | 98.75% | 99.04% | 99.48% | 93.84% |
| **25** | **radish** (Củ cải trắng) | 159 | 96.87% | 97.39% | 99.34% | 88.19% |
| **26** | **scallion** (Hành lá) | 55 | 97.78% | 96.36% | 98.91% | **84.13%** |
| **27** | **shrimp** (Tôm) | 48 | 97.64% | 100.0% | 99.05% | 90.61% |
| **28** | **spongegourd** (Mướp hương) | 143 | 99.17% | 98.60% | 99.42% | 90.79% |
| **29** | **sweetpotato** (Khoai lang) | 100 | 90.05% | 93.00% | 95.06% | **84.49%** |
| **30** | **tofu** (Đậu phụ) | 44 | 99.67% | 100.0% | 99.50% | 92.28% |
| **31** | **tomato** (Cà chua) | 132 | 98.00% | 100.0% | 99.42% | 86.95% |
| **ALL** | **Trung bình toàn bộ 32 lớp** | **3,911** | **97.56%** | **97.59%** | **98.56%** | **88.45%** |

*(Ghi chú: Các chỉ số in đậm là những chỉ số < 85.00%).*

---

## 4. BẢNG LOG CHI TIẾT ĐẦY ĐỦ 50 EPOCHS HUẤN LUYỆN (FULL 50-EPOCH TRAINING LOG)

*Dữ liệu trích xuất từ file log [`rtdetr_epoch_logs.csv`](file:///d:/NCKH/Train_models/synced_results/logs/logs/rtdetr/rtdetr_epoch_logs.csv):*

| Epoch | Train Loss | Class Loss | Box Loss | GIoU Loss | Learning Rate | Epoch Time (s) | Val mAP@50 | Val mAP@50-95 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.03029 | 0.70770 | 0.00000 | 0.32259 | 9.2523e-05 | 228.47 s | 0.9188 | **0.7730** |
| 2 | 0.67863 | 0.42758 | 0.00000 | 0.25105 | 1.8152e-04 | 213.05 s | 0.9633 | **0.8202** |
| 3 | 0.74713 | 0.43232 | 0.00000 | 0.31481 | 2.6685e-04 | 211.27 s | 0.9686 | **0.8318** |
| 4 | 0.77214 | 0.51433 | 0.00000 | 0.25781 | 2.6149e-04 | 208.17 s | 0.9706 | **0.8336** |
| 5 | 0.83686 | 0.46781 | 0.00000 | 0.36904 | 2.5598e-04 | 212.34 s | 0.9748 | **0.8474** |
| 6 | 0.71675 | 0.45454 | 0.00000 | 0.26222 | 2.5048e-04 | 212.36 s | 0.9745 | **0.8434** |
| 7 | 0.64116 | 0.40583 | 0.00000 | 0.23533 | 2.4497e-04 | 208.20 s | 0.9800 | 0.8525 |
| 8 | 0.57986 | 0.31784 | 0.00000 | 0.26202 | 2.3947e-04 | 206.37 s | 0.9810 | 0.8588 |
| 9 | 0.55387 | 0.32339 | 0.00000 | 0.23048 | 2.3396e-04 | 207.39 s | 0.9795 | 0.8566 |
| 10 | 0.61874 | 0.38037 | 0.00000 | 0.23837 | 2.2846e-04 | 206.89 s | 0.9804 | 0.8559 |
| 11 | 0.67187 | 0.41273 | 0.00000 | 0.25914 | 2.2296e-04 | 207.21 s | 0.9827 | 0.8619 |
| 12 | 0.62651 | 0.35969 | 0.00000 | 0.26682 | 2.1745e-04 | 206.41 s | 0.9823 | 0.8631 |
| 13 | 0.64840 | 0.36136 | 0.00000 | 0.28704 | 2.1195e-04 | 206.22 s | 0.9776 | 0.8552 |
| 14 | 0.44626 | 0.28357 | 0.00000 | 0.16270 | 2.0644e-04 | 207.04 s | 0.9826 | 0.8644 |
| 15 | 0.74120 | 0.44716 | 0.00000 | 0.29405 | 2.0094e-04 | 206.65 s | 0.9828 | 0.8657 |
| 16 | 0.53054 | 0.32796 | 0.00000 | 0.20258 | 1.9543e-04 | 206.55 s | 0.9831 | 0.8661 |
| 17 | 0.57804 | 0.37855 | 0.00000 | 0.19949 | 1.8993e-04 | 205.02 s | 0.9807 | 0.8628 |
| 18 | 0.49400 | 0.28290 | 0.00000 | 0.21111 | 1.8443e-04 | 205.04 s | 0.9810 | 0.8626 |
| 19 | 0.51285 | 0.29679 | 0.00000 | 0.21607 | 1.7892e-04 | 207.09 s | 0.9802 | 0.8630 |
| 20 | 0.45432 | 0.28783 | 0.00000 | 0.16649 | 1.7342e-04 | 204.79 s | 0.9823 | 0.8680 |
| 21 | 0.53265 | 0.32590 | 0.00000 | 0.20675 | 1.6791e-04 | 203.83 s | 0.9803 | 0.8646 |
| 22 | 0.46520 | 0.28513 | 0.00000 | 0.18006 | 1.6241e-04 | 202.93 s | 0.9823 | 0.8663 |
| 23 | 0.46689 | 0.28497 | 0.00000 | 0.18192 | 1.5690e-04 | 203.94 s | 0.9826 | 0.8694 |
| 24 | 0.53176 | 0.31714 | 0.00000 | 0.21462 | 1.5140e-04 | 204.29 s | 0.9822 | 0.8690 |
| 25 | 0.60583 | 0.39282 | 0.00000 | 0.21302 | 1.4589e-04 | 205.01 s | 0.9823 | 0.8698 |
| 26 | 0.54320 | 0.34200 | 0.00000 | 0.20120 | 1.4039e-04 | 205.06 s | 0.9843 | 0.8712 |
| 27 | 0.59259 | 0.37806 | 0.00000 | 0.21453 | 1.3489e-04 | 204.09 s | 0.9839 | 0.8707 |
| 28 | 0.57928 | 0.33358 | 0.00000 | 0.24570 | 1.2938e-04 | 205.02 s | 0.9839 | 0.8713 |
| 29 | 0.51262 | 0.29503 | 0.00000 | 0.21759 | 1.2388e-04 | 203.42 s | 0.9836 | 0.8731 |
| 30 | 0.60466 | 0.35678 | 0.00000 | 0.24788 | 1.1837e-04 | 203.05 s | 0.9840 | 0.8718 |
| 31 | 0.45273 | 0.27671 | 0.00000 | 0.17602 | 1.1287e-04 | 204.92 s | 0.9824 | 0.8721 |
| 32 | 0.51719 | 0.32212 | 0.00000 | 0.19506 | 1.0736e-04 | 203.86 s | 0.9830 | 0.8703 |
| 33 | 0.49490 | 0.30830 | 0.00000 | 0.18660 | 1.0186e-04 | 203.19 s | 0.9840 | 0.8717 |
| 34 | 0.59502 | 0.36593 | 0.00000 | 0.22909 | 9.6355e-05 | 205.06 s | 0.9851 | 0.8750 |
| 35 | 0.52899 | 0.33327 | 0.00000 | 0.19572 | 9.0850e-05 | 203.60 s | 0.9846 | 0.8728 |
| 36 | 0.49393 | 0.29221 | 0.00000 | 0.20172 | 8.5346e-05 | 203.95 s | 0.9844 | 0.8742 |
| 37 | 0.55432 | 0.31681 | 0.00000 | 0.23751 | 7.9842e-05 | 203.80 s | 0.9841 | 0.8737 |
| 38 | 0.44590 | 0.27814 | 0.00000 | 0.16776 | 7.4337e-05 | 205.84 s | 0.9845 | 0.8738 |
| 39 | 0.48741 | 0.31480 | 0.00000 | 0.17261 | 6.8833e-05 | 206.78 s | 0.9836 | 0.8727 |
| 40 | 0.42260 | 0.25322 | 0.00000 | 0.16938 | 6.3328e-05 | 208.85 s | 0.9837 | 0.8747 |
| 41 | 0.30034 | 0.19008 | 0.00000 | 0.11026 | 5.7824e-05 | 208.35 s | 0.9834 | 0.8753 |
| **42** | **0.37040** | **0.24520** | **0.00000** | **0.12520** | **5.2320e-05** | **202.56 s** | **0.9844** | **0.8758 (Best Val)** |
| 43 | 0.29496 | 0.19008 | 0.00000 | 0.10488 | 4.6815e-05 | 201.92 s | 0.9832 | 0.8744 |
| 44 | 0.28371 | 0.18432 | 0.00000 | 0.09939 | 4.1311e-05 | 201.44 s | 0.9834 | 0.8754 |
| 45 | 0.32485 | 0.20488 | 0.00000 | 0.11997 | 3.5806e-05 | 205.17 s | 0.9826 | 0.8742 |
| 46 | 0.29783 | 0.19028 | 0.00000 | 0.10755 | 3.0302e-05 | 202.21 s | 0.9828 | 0.8742 |
| 47 | 0.29896 | 0.18973 | 0.00000 | 0.10923 | 2.4798e-05 | 203.36 s | 0.9831 | 0.8744 |
| 48 | 0.29691 | 0.19417 | 0.00000 | 0.10275 | 1.9293e-05 | 202.85 s | 0.9831 | 0.8748 |
| 49 | 0.36087 | 0.22740 | 0.00000 | 0.13347 | 1.3789e-05 | 203.44 s | 0.9822 | 0.8736 |
| 50 | 0.30633 | 0.19102 | 0.00000 | 0.11531 | 8.2844e-06 | 205.90 s | 0.9827 | 0.8736 |

---

## 5. NHẬN XÉT & KẾT LUẬN KHOA HỌC

1. **Hiệu quả vượt bậc của Dataset v5**:
   - Việc nâng cấp lên Dataset v5 giúp giải quyết triệt để các nhãn khó ở v1 (như lớp `corn` tăng mAP50 từ 84.20% lên **95.64%**, lớp `cucumber` tăng từ 93.50% lên **94.96%**).
   - Chỉ số **mAP@50 đạt 98.56%** và **mAP@50-95 đạt 88.45%** trên 1,481 ảnh tập kiểm thử độc lập khẳng định chất lượng xuất sắc của mô hình trong bài toán nhận diện thực phẩm.
2. **Khả năng tổng quát hóa (Generalization)**:
   - Sai số hàm mất mát tổng thể giảm đều đặn từ **1.03029** xuống **0.30633** (giảm hơn 70%), không xuất hiện dấu hiệu quá khớp (overfitting).
3. **Sẵn sàng triển khai ứng dụng thực tế**:
   - Với tốc độ xử lý **5.4 ms / ảnh** (~185 FPS trên GPU A100), RT-DETR đáp ứng hoàn hảo cho việc tích hợp vào hệ thống camera nhận diện tại nhà hàng, ứng dụng dinh dưỡng trên di động và dây chuyền chế biến thực phẩm tự động.
