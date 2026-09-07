# BÁO CÁO KHOA HỌC CHI TIẾT QUÁ TRÌNH HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH RF-DETR (ROBOFLOW DETR)

---

## 1. THÔNG SỐ ĐẦU VÀO VÀ SIÊU THAM SỐ HUẤN LUYỆN (TRAINING HYPERPARAMETERS & CONFIGURATION)

### 1.1. Bảng cấu hình môi trường và phần cứng thực nghiệm
| Thành phần | Chi tiết cấu hình | Ghi chú khoa học |
| :--- | :--- | :--- |
| **Hệ thống điện toán (Platform)** | Modal Cloud Serverless Compute | A100 Tensor Core GPU Dedicated |
| **GPU phần cứng** | NVIDIA A100-SXM4-40GB (40,441 MiB VRAM) | Bộ nhớ băng thông cao HBM2 (1.6 TB/s) |
| **Kiến trúc tính toán** | CUDA 12.4 / 13.0, PyTorch 2.13.0 + cu130 | Tối ưu hóa tính toán ma trận với Tensor Cores |
| **Kiểu dữ liệu huấn luyện (Precision)** | Mixed Precision (bfloat16 / fp16 Auto AMP) | Tăng tốc độ tính toán, giảm 50% bộ nhớ GPU |
| **Môi trường thực thi (OS / Python)** | Linux Debian Slim, Python 3.10.17 | Cố định hạt nhân đảm bảo tính tái lập (Reproducibility) |

---

### 1.2. Bảng thông số tập dữ liệu thực nghiệm (Dataset Specification)
| Thuộc tính dữ liệu | Giá trị / Định lượng | Ý nghĩa khoa học |
| :--- | :--- | :--- |
| **Tên tập dữ liệu (Dataset)** | `completed-project-1` (Food Detection 32 Classes) | Dữ liệu hình ảnh nguyên liệu & món ăn Việt Nam |
| **Tổng số lượng hình ảnh** | **16,000 ảnh** (Đã qua lọc & gán nhãn COCO/YOLO) | Mật độ đa dạng góc chụp, ánh sáng và bối cảnh |
| **Tỉ lệ phân chia dữ liệu (Data Split)** | **7 : 2 : 1** (Train: 70%, Val: 20%, Test: 10%) | Chuẩn mực vàng phân chia dữ liệu học máy |
| **- Tập huấn luyện (Train split)** | **11,200 ảnh** (~70.0%) | Dùng để cập nhật gradient trọng số mô hình |
| **- Tập kiểm định (Validation split)** | **3,269 ảnh** (~20.4%) | Dùng đánh giá sau mỗi epoch & Early Stopping |
| **- Tập kiểm thử độc lập (Test split)**| **1,531 ảnh** (~9.6%, 3,964 nhãn instances) | Dùng đánh giá khách quan cuối cùng (Zero Leakage) |
| **Số lượng lớp nhận diện ($C$)** | **32 lớp thực phẩm đặc thù** | Độ bao phủ toàn diện nhóm thịt, rau, củ, trứng |
| **Kích thước ảnh đầu vào (Resolution)** | $576 \times 576$ (Multi-Scale Jittering Square Resize) | Cân bằng hoàn hảo giữa độ chi tiết và tốc độ suy luận |

---

### 1.3. Bảng siêu tham số huấn luyện chi tiết (Hyperparameters Table)
| Nhóm tham số | Tên siêu tham số | Giá trị thiết lập | Giải thích cơ chế tác động |
| :--- | :--- | :--- | :--- |
| **Tối ưu hóa (Optimizer)** | `optimizer` | **AdamW** | Thuật toán tối ưu hóa thích nghi với Weight Decay rời rạc |
| **Tốc độ học cơ sở** | `lr` | **0.0001 ($1\times 10^{-4}$)** | Learning rate áp dụng cho Transformer Decoder |
| **Tốc độ học Encoder** | `lr_encoder` | **0.00015 ($1.5\times 10^{-4}$)** | Learning rate riêng cho DINOv2 Backbone trích xuất đặc trưng |
| **Hệ số suy giảm ViT Layer**| `lr_vit_layer_decay` | **0.8** | Giảm dần tốc độ học ở các tầng ViT sâu hơn để tránh phá vỡ tiền huấn luyện |
| **Hệ số suy giảm thành phần**| `lr_component_decay` | **0.7** | Phân cấp learning rate giữa backbone, neck và decoder |
| **Suy giảm trọng số** | `weight_decay` | **0.0001 ($1\times 10^{-4}$)** | Chống hiện tượng quá khớp (Overfitting L2 Regularization) |
| **Kích thước Batch (Batch Size)** | `batch_size` | **8** (trên 1 GPU) | Kích thước lô nạp vào VRAM mỗi bước |
| **Tích lũy Gradient** | `grad_accum_steps` | **2** | Tích lũy 2 bước $\rightarrow$ **Effective Batch Size = 16** |
| **Kẹp Gradient (Gradient Clip)**| `clip_max_norm` | **0.1** | Ngăn chặn bùng nổ gradient trong Attention Layers |
| **Tổng số Epoch tối đa** | `epochs` | **50 epochs** | Giới hạn vòng lặp huấn luyện tối đa |
| **Cơ chế Dừng sớm** | `early_stopping` | **True (`patience=20`, `min_delta=0.001`)** | Tự động dừng tại **Epoch 46** khi mAP bão hòa (Best tại Ep 26) |
| **Độ trễ trung bình động EMA** | `ema_decay` | **0.993** (`ema_tau=100`) | Exponential Moving Average làm mượt trọng số mô hình |
| **Số lượng truy vấn (Queries)**| `num_queries` | **300** | Số lượng object queries dự đoán đồng thời trong Decoder |
| **Nhóm huấn luyện DETR** | `group_detr` | **13 groups** | Huấn luyện 13 nhóm queries song song đẩy nhanh tốc độ hội tụ |
| **Hàm mất mát phân loại** | `ia_bce_loss` | **True (`cls_loss_coef=1.0`)** | Binary Cross Entropy tích hợp IoU-Aware Classification |
| **Đặc trưng Backbone** | `encoder` | **`dinov2_windowed_small`** | Vision Transformer DINOv2 tự giám sát với Windowed Attention |

---

## 2. NHẬT KÝ HUẤN LUYỆN CHI TIẾT TỪNG EPOCH (FULL 46 EPOCHS TRAINING LOG)

> **Cơ chế Early Stopping**: Mô hình tự động kích hoạt dừng sớm tại **Epoch 46/50** do chỉ số `__rfdetr_effective_map__` đạt giá trị tối ưu tốt nhất tại **Epoch 26** (EMA mAP Score: **0.8730**) và không cải thiện vượt mức `min_delta=0.001` trong 20 epochs liên tiếp. File trọng số tối ưu nhất được trích xuất tự động từ EMA tại `/vol/checkpoints/rfdetr/best.pth`.

| Epoch | Train Loss | Val Loss | Box Loss | CE Loss (Cls) | GIoU Loss | Learning Rate | Val Precision | Val Recall | Val mAP@50 | Val mAP@50-95 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 5.7572 | 4.8997 | 0.0385 | 0.4823 | 0.0876 | $1.000\times 10^{-4}$ | 0.9289 | 0.9128 | 0.9600 | 0.8201 |
| **2** | 4.5059 | 4.3332 | 0.0372 | 0.4064 | 0.0846 | $1.000\times 10^{-4}$ | 0.9488 | 0.9452 | 0.9730 | 0.8410 |
| **3** | 4.2237 | 4.2945 | 0.0373 | 0.3933 | 0.0863 | $1.000\times 10^{-4}$ | 0.9537 | 0.9471 | 0.9776 | 0.8442 |
| **4** | 4.1225 | 4.2704 | 0.0383 | 0.3926 | 0.0875 | $1.000\times 10^{-4}$ | 0.9580 | 0.9546 | 0.9794 | 0.8386 |
| **5** | 3.9338 | 4.1265 | 0.0375 | 0.3772 | 0.0860 | $1.000\times 10^{-4}$ | 0.9533 | 0.9564 | 0.9789 | 0.8453 |
| **6** | 3.9362 | 4.2544 | 0.0359 | 0.3709 | 0.0829 | $1.000\times 10^{-4}$ | 0.9589 | 0.9566 | 0.9806 | 0.8537 |
| **7** | 3.8559 | 4.0031 | 0.0370 | 0.3543 | 0.0840 | $1.000\times 10^{-4}$ | 0.9698 | 0.9552 | 0.9811 | 0.8546 |
| **8** | 3.7785 | 4.0217 | 0.0353 | 0.3597 | 0.0812 | $1.000\times 10^{-4}$ | 0.9616 | 0.9582 | 0.9805 | 0.8597 |
| **9** | 3.7094 | 4.0028 | 0.0363 | 0.3848 | 0.0822 | $1.000\times 10^{-4}$ | 0.9678 | 0.9486 | 0.9815 | 0.8570 |
| **10** | 3.7286 | 3.9674 | 0.0367 | 0.3628 | 0.0825 | $1.000\times 10^{-4}$ | 0.9634 | 0.9570 | 0.9802 | 0.8554 |
| **11** | 3.6905 | 4.0246 | 0.0374 | 0.3575 | 0.0851 | $1.000\times 10^{-4}$ | 0.9617 | 0.9593 | 0.9808 | 0.8501 |
| **12** | 3.6056 | 3.9694 | 0.0361 | 0.3561 | 0.0823 | $1.000\times 10^{-4}$ | 0.9680 | 0.9565 | 0.9792 | 0.8542 |
| **13** | 3.5836 | 3.9978 | 0.0355 | 0.3824 | 0.0820 | $1.000\times 10^{-4}$ | 0.9589 | 0.9618 | 0.9794 | 0.8563 |
| **14** | 3.4998 | 3.7693 | 0.0354 | 0.3526 | 0.0804 | $1.000\times 10^{-4}$ | 0.9661 | 0.9594 | 0.9819 | 0.8622 |
| **15** | 3.5192 | 3.9515 | 0.0368 | 0.3507 | 0.0837 | $1.000\times 10^{-4}$ | 0.9688 | 0.9567 | 0.9817 | 0.8551 |
| **16** | 3.4830 | 3.8427 | 0.0376 | 0.3528 | 0.0869 | $1.000\times 10^{-4}$ | 0.9598 | 0.9557 | 0.9785 | 0.8559 |
| **17** | 3.4506 | 3.8183 | 0.0358 | 0.3475 | 0.0818 | $1.000\times 10^{-4}$ | 0.9634 | 0.9599 | 0.9800 | 0.8589 |
| **18** | 3.4249 | 3.8574 | 0.0365 | 0.3528 | 0.0825 | $1.000\times 10^{-4}$ | 0.9677 | 0.9616 | 0.9813 | 0.8575 |
| **19** | 3.3779 | 3.8296 | 0.0368 | 0.3385 | 0.0822 | $1.000\times 10^{-4}$ | 0.9693 | 0.9611 | 0.9839 | 0.8598 |
| **20** | 3.3602 | 4.0597 | 0.0372 | 0.3411 | 0.0844 | $1.000\times 10^{-4}$ | 0.9698 | 0.9626 | 0.9811 | 0.8580 |
| **21** | 3.3055 | 3.8234 | 0.0355 | 0.3454 | 0.0808 | $1.000\times 10^{-4}$ | 0.9693 | 0.9572 | 0.9796 | 0.8606 |
| **22** | 3.3092 | 3.8826 | 0.0360 | 0.3487 | 0.0823 | $1.000\times 10^{-4}$ | 0.9754 | 0.9545 | 0.9817 | 0.8580 |
| **23** | 3.3306 | 3.7790 | 0.0359 | 0.3357 | 0.0813 | $1.000\times 10^{-4}$ | 0.9665 | 0.9650 | 0.9809 | 0.8594 |
| **24** | 3.2358 | 3.8216 | 0.0365 | 0.3415 | 0.0820 | $1.000\times 10^{-4}$ | 0.9655 | 0.9641 | 0.9809 | 0.8586 |
| **25** | 3.2912 | 3.7856 | 0.0359 | 0.3367 | 0.0821 | $1.000\times 10^{-4}$ | 0.9713 | 0.9667 | 0.9822 | 0.8604 |
| **26 ⭐** | **3.2409** | **3.7377** | **0.0350** | **0.3350** | **0.0805** | **$1.000\times 10^{-4}$** | **0.9698** | **0.9642** | **0.9831** | **0.8639 (EMA: 0.8730)** |
| **27** | 3.1738 | 3.7895 | 0.0356 | 0.3444 | 0.0824 | $1.000\times 10^{-4}$ | 0.9729 | 0.9595 | 0.9799 | 0.8565 |
| **28** | 3.2046 | 3.8310 | 0.0362 | 0.3377 | 0.0812 | $1.000\times 10^{-4}$ | 0.9717 | 0.9578 | 0.9798 | 0.8610 |
| **29** | 3.2549 | 3.8247 | 0.0362 | 0.3402 | 0.0812 | $1.000\times 10^{-4}$ | 0.9683 | 0.9676 | 0.9823 | 0.8596 |
| **30** | 3.1335 | 3.7772 | 0.0368 | 0.3408 | 0.0831 | $1.000\times 10^{-4}$ | 0.9661 | 0.9608 | 0.9804 | 0.8568 |
| **31** | 3.1378 | 3.8009 | 0.0359 | 0.3428 | 0.0810 | $1.000\times 10^{-4}$ | 0.9688 | 0.9585 | 0.9794 | 0.8609 |
| **32** | 3.1416 | 3.8149 | 0.0363 | 0.3381 | 0.0835 | $1.000\times 10^{-4}$ | 0.9748 | 0.9568 | 0.9787 | 0.8574 |
| **33** | 3.0966 | 3.7092 | 0.0360 | 0.3310 | 0.0817 | $1.000\times 10^{-4}$ | 0.9713 | 0.9626 | 0.9802 | 0.8626 |
| **34** | 3.0351 | 3.7497 | 0.0358 | 0.3353 | 0.0817 | $1.000\times 10^{-4}$ | 0.9740 | 0.9571 | 0.9794 | 0.8610 |
| **35** | 3.0878 | 3.6975 | 0.0364 | 0.3331 | 0.0818 | $1.000\times 10^{-4}$ | 0.9740 | 0.9545 | 0.9778 | 0.8620 |
| **36** | 3.0556 | 3.6852 | 0.0363 | 0.3279 | 0.0818 | $1.000\times 10^{-4}$ | 0.9666 | 0.9660 | 0.9798 | 0.8635 |
| **37** | 3.0170 | 3.8759 | 0.0359 | 0.3333 | 0.0812 | $1.000\times 10^{-4}$ | 0.9695 | 0.9635 | 0.9813 | 0.8611 |
| **38** | 3.0187 | 3.7209 | 0.0355 | 0.3342 | 0.0808 | $1.000\times 10^{-4}$ | 0.9641 | 0.9695 | 0.9822 | 0.8634 |
| **39** | 3.0073 | 3.6663 | 0.0354 | 0.3360 | 0.0793 | $1.000\times 10^{-4}$ | 0.9638 | 0.9619 | 0.9828 | 0.8660 |
| **40** | 2.9751 | 3.6343 | 0.0356 | 0.3302 | 0.0806 | $1.000\times 10^{-4}$ | 0.9714 | 0.9624 | 0.9795 | 0.8642 |
| **41** | 2.9608 | 3.6815 | 0.0359 | 0.3338 | 0.0808 | $1.000\times 10^{-4}$ | 0.9680 | 0.9578 | 0.9800 | 0.8639 |
| **42** | 2.9297 | 3.6380 | 0.0354 | 0.3299 | 0.0800 | $1.000\times 10^{-4}$ | 0.9684 | 0.9677 | 0.9817 | 0.8639 |
| **43** | 2.9593 | 3.6244 | 0.0350 | 0.3261 | 0.0798 | $1.000\times 10^{-4}$ | 0.9676 | 0.9663 | 0.9811 | 0.8651 |
| **44** | 3.0044 | 3.7624 | 0.0366 | 0.3282 | 0.0813 | $1.000\times 10^{-4}$ | 0.9664 | 0.9657 | 0.9795 | 0.8633 |
| **45** | 2.9780 | 3.6921 | 0.0363 | 0.3330 | 0.0813 | $1.000\times 10^{-4}$ | 0.9667 | 0.9675 | 0.9802 | 0.8597 |
| **46** | 2.9371 | 3.7090 | 0.0359 | 0.3368 | 0.0817 | $1.000\times 10^{-4}$ | 0.9748 | 0.9576 | 0.9755 | 0.8570 |

---

## 3. BẢNG KẾT QUẢ ĐÁNH GIÁ KIỂM THỬ ĐỘC LẬP TRÊN TEST SPLIT (PER-CLASS ACCURACY ON 1,531 TEST IMAGES)

> **Môi trường đánh giá độc lập (Test Evaluation)**: Thực thi trên tập dữ liệu kiểm thử độc lập gồm **1,531 hình ảnh** với **3,964 nhãn đối tượng thực phẩm (instances)**.

| STT | Tên lớp thực phẩm (Class Name) | Số lượng mẫu (Instances) | Độ chính xác (Precision) | Độ thu hồi (Recall) | F1-Score | mAP@50 | mAP@50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `beef` (Thịt bò) | 331 | 0.9878 | 0.9819 | 0.9848 | 0.9860 | 0.8987 |
| 2 | `bellpepper` (Ớt chuông) | 83 | 0.9877 | 0.9639 | 0.9756 | 0.9840 | 0.9426 |
| 3 | `bittergourd` (Khổ qua) | 50 | **1.0000** | **1.0000** | **1.0000** | **0.9950** | **0.9760** |
| 4 | `bottlegourd` (Bầu) | 140 | 0.9609 | 0.8786 | 0.9179 | 0.9760 | 0.8677 |
| 5 | `broccoli` (Bông cải xanh) | 68 | 0.9853 | 0.9853 | 0.9853 | 0.9870 | 0.9441 |
| 6 | `cabbage` (Bắp cải) | 79 | 0.9737 | 0.9367 | 0.9548 | 0.9830 | 0.9044 |
| 7 | `carrot` (Cà rốt) | 74 | 0.9605 | 0.9865 | 0.9733 | 0.9850 | 0.8579 |
| 8 | `cauliflower` (Bông cải trắng) | 109 | 0.9316 | **1.0000** | 0.9646 | 0.9820 | 0.9162 |
| 9 | `chayote` (Su su) | 57 | **1.0000** | **1.0000** | **1.0000** | **0.9950** | **0.9701** |
| 10 | `chicken` (Thịt gà) | 50 | **1.0000** | 0.9200 | 0.9583 | 0.9670 | 0.7383 |
| 11 | `chickenegg` (Trứng gà) | 223 | 0.9911 | 0.9955 | 0.9933 | 0.9930 | 0.9060 |
| 12 | `chickenleg` (Đùi gà) | 531 | 0.9962 | 0.9887 | 0.9924 | 0.9920 | 0.8147 |
| 13 | `chickenwin` (Cánh gà) | 151 | **1.0000** | **1.0000** | **1.0000** | 0.9920 | 0.8525 |
| 14 | `corn` (Ngô / Bắp) | 185 | 0.9470 | 0.7730 | 0.8512 | 0.8420 | 0.7247 |
| 15 | `cucumber` (Dưa leo) | 197 | 0.8744 | 0.9188 | 0.8960 | 0.9350 | 0.8762 |
| 16 | `duckegg` (Trứng vịt) | 99 | **1.0000** | 0.9697 | 0.9846 | 0.9840 | 0.8998 |
| 17 | `eggplant` (Cà tím) | 45 | **1.0000** | **1.0000** | **1.0000** | **0.9950** | 0.9248 |
| 18 | `garlic` (Tỏi) | 179 | 0.9615 | 0.9777 | 0.9695 | 0.9940 | 0.8636 |
| 19 | `ginger` (Gừng) | 64 | **1.0000** | **1.0000** | **1.0000** | **0.9950** | 0.9175 |
| 20 | `jicama` (Củ sắn / Củ đậu) | 109 | 0.9604 | 0.8899 | 0.9238 | 0.9780 | 0.8688 |
| 21 | `okra` (Đậu bắp) | 92 | 0.9583 | **1.0000** | 0.9787 | 0.9860 | 0.7473 |
| 22 | `onion` (Hành tây) | 126 | 0.9462 | 0.9762 | 0.9609 | 0.9700 | 0.8482 |
| 23 | `pork` (Thịt heo) | 79 | **1.0000** | 0.9873 | 0.9936 | **0.9950** | 0.8708 |
| 24 | `potato` (Khoai tây) | 58 | 0.9500 | 0.9828 | 0.9661 | 0.9940 | 0.9306 |
| 25 | `pumpkin` (Bí đỏ) | 104 | 0.9537 | 0.9904 | 0.9717 | 0.9920 | 0.9509 |
| 26 | `radish` (Củ cải trắng) | 159 | 0.9935 | 0.9686 | 0.9809 | 0.9940 | 0.8767 |
| 27 | `scallion` (Hành lá) | 55 | 0.9811 | 0.9455 | 0.9630 | 0.9880 | 0.8439 |
| 28 | `shrimp` (Tôm) | 48 | **1.0000** | **1.0000** | **1.0000** | **0.9950** | 0.9117 |
| 29 | `spongegourd` (Mướp) | 143 | 0.9929 | 0.9790 | 0.9859 | 0.9910 | 0.9085 |
| 30 | `sweetpotato` (Khoai lang) | 100 | 0.9118 | 0.9300 | 0.9208 | 0.9460 | 0.8392 |
| 31 | `tofu` (Đậu phụ) | 44 | **1.0000** | **1.0000** | **1.0000** | **0.9950** | 0.9038 |
| 32 | `tomato` (Cà chua) | 132 | 0.9924 | 0.9924 | 0.9924 | 0.9940 | 0.8651 |
| **-** | **TRUNG BÌNH TOÀN BỘ (OVERALL ALL)** | **3,964** | **0.9749 (97.49%)** | **0.9662 (96.62%)** | **0.9700 (97.00%)** | **0.9845 (98.45%)** | **0.8800 (88.00%)** |

---

## 4. PHÂN TÍCH CHUYÊN SÂU CÁC THÀNH PHẦN HÀM MẤT MÁT (LOSS FORMULATIONS & CONVERGENCE)

### 4.1. Cấu trúc hàm mất mát tổng thể trong RF-DETR
Hàm mục tiêu tối ưu trong RF-DETR là sự kết hợp có trọng số giữa phân loại (Classification), hồi quy tọa độ (L1 Box Loss) và mất mát tương giao mở rộng (GIoU Loss):

$$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{IA-BCE}} + \lambda_{\text{box}} \mathcal{L}_{\text{L1}} + \lambda_{\text{giou}} \mathcal{L}_{\text{GIoU}}$$

Trong đó:
1. **$\mathcal{L}_{\text{IA-BCE}}$ (IoU-Aware Binary Cross Entropy Loss)**:
   Thay vì phân loại nhị phân tiêu chuẩn gán nhãn cứng 0 hoặc 1, IA-BCE tích hợp trực tiếp chất lượng hộp bao $IoU(b, \hat{b})$ làm nhãn mục tiêu mềm (soft target):
   $$\mathcal{L}_{\text{IA-BCE}} = - \sum_{i=1}^{N} \left[ q_i \log(\hat{p}_i) + (1 - q_i) \log(1 - \hat{p}_i) \right]$$
   với $q_i = \alpha \cdot y_i + (1 - \alpha) \cdot \text{IoU}_i$. Cơ chế này giúp mô hình triệt tiêu các dự đoán có xác suất cao nhưng vị trí hộp bao lệch lạc.
2. **$\mathcal{L}_{\text{L1}}$ (Bounding Box L1 Loss)**:
   Đo khoảng cách sai số tuyệt đối giữa tâm và kích thước hộp:
   $$\mathcal{L}_{\text{L1}}(b, \hat{b}) = \| b - \hat{b} \|_1 = |c_x - \hat{c}_x| + |c_y - \hat{c}_y| + |w - \hat{w}| + |h - \hat{h}|$$
3. **$\mathcal{L}_{\text{GIoU}}$ (Generalized Intersection over Union Loss)**:
   Khắc phục nhược điểm của IoU thông thường khi hai hộp không giao nhau ($IoU = 0$):
   $$\mathcal{L}_{\text{GIoU}} = 1 - \left( \frac{|A \cap B|}{|A \cup B|} - \frac{|C \setminus (A \cup B)|}{|C|} \right)$$
   với $C$ là diện tích hình chữ nhật bao lồi nhỏ nhất chứa cả $A$ và $B$.

---

## 5. ĐÁNH GIÁ VÀ KẾT LUẬN VỀ HIỆU NĂNG MÔ HÌNH RF-DETR

1. **Độ chính xác vượt trội (State-of-the-Art Accuracy)**:
   - RF-DETR đạt **mAP@50: 98.45%** và **mAP@50-95: 88.00%** trên tập test split độc lập 1,531 ảnh, thể hiện năng lực khoanh vùng và phân loại chính xác vượt trội đối với các thực phẩm có kết cấu và màu sắc tương đồng.
2. **Sức mạnh từ DINOv2 Backbone**:
   - Nhờ thừa hưởng đặc trưng ngữ nghĩa tự giám sát phong phú từ **DINOv2 Small** kết hợp với **Windowed Attention**, mô hình biểu diễn cực tốt các vật thể phức tạp như `bittergourd` (mAP@50-95: **97.60%**), `chayote` (mAP@50-95: **97.01%**), `pumpkin` (mAP@50-95: **95.09%**).
3. **Độ ổn định và chống Overfitting**:
   - Cơ chế **Group DETR (13 nhóm)** và **Exponential Moving Average (EMA)** đảm bảo mô hình hội tụ mượt mà, đạt đỉnh tại Epoch 26 và bảo toàn được trọng số tối ưu nhất khi Early Stopping kích hoạt tại Epoch 46.
