# BÁO CÁO KHOA HỌC ĐỐI ĐẦU VÀ SO SÁNH TOÀN DIỆN: RT-DETR VS RF-DETR (PHIÊN BẢN v4)
### NGHIÊN CỨU THỰC NGHIỆM TRÊN TẬP DỮ LIỆU NHẬN DIỆN MÓN ĂN & NGUYÊN LIỆU VIỆT NAM (32 LỚP)

---

## 1. TỔNG QUAN THỰC NGHIỆM & QUY CHUẨN ĐỒNG BỘ v4

Nghiên cứu thực nghiệm này đánh giá khách quan, độc lập và đồng bộ 100% (Head-to-Head Fair Benchmark) giữa hai kiến trúc phát hiện đối tượng Transformer thời gian thực tiên tiến nhất hiện nay:
1. **RT-DETR (Real-Time DEtection TRansformer - Phiên bản RT-DETR-L)** phát triển bởi Baidu / Ultralytics.
2. **RF-DETR (Roboflow DETR - Phiên bản RF-DETR Medium)** phát triển bởi Roboflow Research tích hợp mô hình nền tảng thị giác tự giám sát **DINOv2** của Meta AI.

### Quy chuẩn đồng bộ phiên bản v4:
- **Tập dữ liệu chuẩn**: Roboflow `completed-project-5` gồm **16,000 hình ảnh**, chia tỉ lệ 7:2:1 (Train: 10,352 ảnh, Val: 2,976 ảnh, Test: 1,481 ảnh với 3,911 nhãn đối tượng) trên **32 lớp thực phẩm**.
- **Kích thước ảnh đồng bộ**: $640 \times 640$ (Square Resize) cho cả huấn luyện và kiểm thử.
- **Phần cứng đồng bộ**: **1x NVIDIA A100-SXM4-40GB GPU** (Modal Cloud Serverless Compute).
- **Siêu tham số đồng bộ**: `batch_size = 16`, `patience = 10`, `max_epochs = 80`, optimizer AdamW.
- **Giao thức đánh giá (Evaluation Protocol)**:
  - **mAP@50** và **mAP@50-95**: Đánh giá tại `conf = 0.01` theo chuẩn COCO benchmark quốc tế với bước nhảy IoU step $0.05$ (10 mức: $0.50 : 0.05 : 0.95$).
  - **Ma trận nhầm lẫn (Confusion Matrix)**: Đánh giá tại ngưỡng hoạt động tối ưu `conf = 0.25` và IoU threshold $0.50$, cấu hình hiển thị số liệu trực quan (`annot=True`, định dạng 2 chữ số thập phân cho ma trận chuẩn hóa và số nguyên cho ma trận đếm số lượng) để phản ánh chính xác sự nhầm lẫn giữa các lớp, loại bỏ nhiễu nền.
  - **Hardware Benchmark**: Đo đạc tham số (Params M), độ phức tạp (GFLOPs tại 640x640), độ trễ suy luận (Latency ms) và thông lượng (Throughput FPS) tại `batch_size = 1` với 20 lượt Warmup và cơ chế `torch.cuda.synchronize()`.

---

## 2. BẢNG SO SÁNH TỔNG THỂ HIỆU NĂNG & THÔNG SỐ KỸ THUẬT (OVERALL BENCHMARK)

*(Toàn bộ chỉ số được trích xuất 100% từ kết quả đánh giá thực nghiệm độc lập trên tập Test gồm 1,481 ảnh)*

| Nhóm chỉ số | Tiêu chí kỹ thuật / Hiệu năng | RT-DETR (RT-DETR-L) | RF-DETR (RF-DETR Medium) | So sánh / Chênh lệch ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **Kiến trúc** | **Mạng xương sống (Backbone)** | HGNetv2 (Hybrid Encoder) | DINOv2 Windowed Small (ViT) | DINOv2 tự giám sát không nhãn |
| | **Số lượng tham số (Parameters)** | **32.87 Triệu (32.87M)** | 33.60 Triệu (33.60M) | Tương đương cân xứng (~2.2%) |
| | **Độ phức tạp tính toán (FLOPs)** | 110.0 GFLOPs | **2.87 GFLOPs** | **RF-DETR nhẹ hơn đáng kể** |
| **Phần cứng A100** | **Độ trễ suy luận (Latency @ batch=1)**| **7.50 ms / ảnh** | 8.20 ms / ảnh | **RT-DETR nhanh hơn ~8.5%** |
| | **Tốc độ khung hình (Throughput)** | **133.3 FPS** | 121.9 FPS | **RT-DETR cao hơn +11.4 FPS** |
| **Huấn luyện** | **Tổng số Epoch chạy thực tế** | 69 Epochs (Early Stopping) | **23 Epochs (Early Stopping)** | **RF-DETR hội tụ nhanh gấp 3 lần** |
| | **Checkpoint tối ưu (Best Epoch)** | Epoch 58 | Epoch 13 | RF-DETR đạt đỉnh cực sớm |
| **Độ chính xác Test**| **Độ chính xác (Precision - P)** | **97.70% (0.9770)** | 97.52% (0.9752) | RT-DETR nhỉnh hơn +0.18% |
| *(conf = 0.01)* | **Độ thu hồi (Recall - R)** | 96.90% (0.9690) | **97.44% (0.9744)** | **RF-DETR cao hơn +0.54% ⭐** |
| | **Điểm F1-Score tổng hợp** | 97.30% (0.9730) | **97.48% (0.9748)** | **RF-DETR cao hơn +0.18% ⭐** |
| | **Độ chính xác mAP@50** | **98.25% (0.9825)** | 96.79% (0.9679) | RT-DETR vượt trội +1.46% |
| | **Độ chính xác vị trí mAP@50-95**| **87.46% (0.8746)** | 86.20% (0.8620) | RT-DETR vượt trội +1.26% |

---

## 3. BẢNG SO SÁNH ĐỐI ĐẦU CHI TIẾT TỪNG LỚP (PER-CLASS COMPARISON ON 1,481 TEST IMAGES)

Dưới đây là bảng đối sánh trực tiếp độ chính xác của 32 lớp thực phẩm trên tập kiểm thử độc lập (Test Split gồm 1,481 ảnh và 3,911 instances thực tế):

| STT | Tên lớp (Class Name) | RT Precision | RF Precision | RT Recall | RF Recall | RT mAP@50 | RF mAP@50 | RT mAP@50-95 | RF mAP@50-95 | Mô hình tối ưu mAP@50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `beef` (Thịt bò) | 0.9878 | 0.9789 | 0.9772 | 0.9789 | 0.9900 | 0.9676 | **0.9049** | 0.8797 | RT-DETR (+2.52%) |
| 2 | `bellpepper` (Ớt chuông) | 0.9537 | 0.9875 | 0.9639 | 0.9518 | 0.9733 | 0.9502 | **0.9275** | 0.9072 | RT-DETR (+2.03%) |
| 3 | `bittergourd` (Khổ qua) | 0.9966 | 1.0000 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9529 | **0.9709** | **RF-DETR (+1.80% ⭐)** |
| 4 | `bottlegourd` (Bầu) | 0.9848 | 0.9485 | 0.9263 | 0.9214 | 0.9668 | 0.9140 | **0.8576** | 0.8223 | RT-DETR (+3.53%) |
| 5 | `broccoli` (Bông cải xanh) | 0.9845 | 0.9706 | 0.9853 | 0.9706 | 0.9819 | 0.9592 | **0.9435** | 0.9083 | RT-DETR (+3.52%) |
| 6 | `cabbage` (Bắp cải) | 0.9732 | 0.9506 | 0.9195 | 0.9747 | 0.9796 | 0.9683 | **0.8950** | 0.8863 | RT-DETR (+0.87%) |
| 7 | `carrot` (Cà rốt) | 0.9780 | 0.9865 | 0.9865 | 0.9865 | 0.9922 | 0.9802 | **0.8645** | 0.8523 | RT-DETR (+1.22%) |
| 8 | `cauliflower` (Bông cải trắng) | 0.9529 | 0.9727 | 1.0000 | 0.9817 | 0.9857 | 0.9732 | **0.9188** | 0.9055 | RT-DETR (+1.33%) |
| 9 | `chayote` (Su su) | 0.9969 | 1.0000 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9565 | **0.9758** | **RF-DETR (+1.93% ⭐)** |
| 10 | `chicken` (Thịt gà) | 0.9659 | 0.9400 | 0.9000 | 0.9400 | 0.9563 | 0.9220 | **0.6821** | 0.6394 | RT-DETR (+4.27%) |
| 11 | `chickenegg` (Trứng gà) | 0.9604 | 0.9821 | 0.9865 | 0.9865 | 0.9928 | 0.9787 | **0.9011** | 0.8861 | RT-DETR (+1.50%) |
| 12 | `chickenleg` (Đùi gà) | 0.9880 | 0.9943 | 0.9849 | 0.9868 | 0.9926 | 0.9802 | **0.8485** | 0.7969 | RT-DETR (+5.16%) |
| 13 | `chickenwin` (Cánh gà) | 0.9983 | 1.0000 | 1.0000 | 0.9934 | 0.9950 | 0.9901 | **0.8871** | 0.8378 | RT-DETR (+4.93%) |
| 14 | `corn` (Ngô / Bắp) | 0.9419 | 0.9542 | 0.9030 | 0.9328 | 0.9539 | 0.9127 | **0.8003** | 0.7775 | RT-DETR (+2.28%) |
| 15 | `cucumber` (Dưa leo) | 0.9410 | 0.9024 | 0.8901 | 0.9391 | 0.9409 | 0.9089 | **0.8372** | 0.8347 | RT-DETR (+0.25%) |
| 16 | `duckegg` (Trứng vịt) | 0.9892 | 0.9796 | 0.9697 | 0.9697 | 0.9801 | 0.9604 | **0.9085** | 0.8920 | RT-DETR (+1.65%) |
| 17 | `eggplant` (Cà tím) | 0.9924 | 1.0000 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9127 | **0.9320** | **RF-DETR (+1.93% ⭐)** |
| 18 | `garlic` (Tỏi) | 0.9887 | 0.9890 | 0.9761 | 1.0000 | 0.9869 | 0.9999 | 0.8551 | **0.8624** | **RF-DETR (+0.73% ⭐)** |
| 19 | `ginger` (Gừng) | 1.0000 | 1.0000 | 0.9983 | 0.9844 | 0.9950 | 0.9802 | **0.9137** | 0.9114 | RT-DETR (+0.23%) |
| 20 | `jicama` (Củ đậu / Củ sắn) | 0.9586 | 0.9720 | 0.8991 | 0.9541 | 0.9739 | 0.9497 | **0.8631** | 0.8502 | RT-DETR (+1.29%) |
| 21 | `okra` (Đậu bắp) | 0.9678 | 0.9684 | 0.9811 | 1.0000 | 0.9757 | 0.9945 | 0.7158 | **0.7326** | **RF-DETR (+1.68% ⭐)** |
| 22 | `onion` (Hành tây) | 0.9529 | 0.9237 | 0.9603 | 0.9603 | 0.9724 | 0.9496 | **0.8191** | 0.8093 | RT-DETR (+0.98%) |
| 23 | `pork` (Thịt heo) | 0.9748 | 1.0000 | 0.9794 | 1.0000 | 0.9842 | 1.0000 | 0.8684 | **0.8783** | **RF-DETR (+0.99% ⭐)** |
| 24 | `potato` (Khoai tây) | 0.9861 | 0.9643 | 1.0000 | 0.9643 | 0.9950 | 0.9604 | **0.9288** | 0.9049 | RT-DETR (+2.39%) |
| 25 | `pumpkin` (Bí đỏ) | 0.9903 | 1.0000 | 0.9864 | 0.9808 | 0.9949 | 0.9802 | **0.9369** | 0.9236 | RT-DETR (+1.33%) |
| 26 | `radish` (Củ cải trắng) | 0.9848 | 0.9871 | 0.9811 | 0.9623 | 0.9944 | 0.9603 | **0.8817** | 0.8395 | RT-DETR (+4.22%) |
| 27 | `scallion` (Hành lá) | 0.9637 | 0.9636 | 0.9660 | 0.9636 | 0.9868 | 0.9584 | 0.7973 | **0.8048** | **RF-DETR (+0.75% ⭐)** |
| 28 | `shrimp` (Tôm) | 0.9971 | 1.0000 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | **0.9130** | 0.9098 | RT-DETR (+0.32%) |
| 29 | `spongegourd` (Mướp) | 0.9919 | 0.9929 | 0.9860 | 0.9790 | 0.9902 | 0.9701 | **0.8950** | 0.8822 | RT-DETR (+1.28%) |
| 30 | `sweetpotato` (Khoai lang) | 0.9483 | 0.9216 | 0.9179 | 0.9400 | 0.9435 | 0.9147 | **0.8279** | 0.8080 | RT-DETR (+1.99%) |
| 31 | `tofu` (Đậu phụ) | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | **0.9085** | 0.9080 | Ngang nhau (~0.05%) |
| 32 | `tomato` (Cà chua) | 0.9774 | 0.9924 | 0.9849 | 0.9924 | 0.9922 | 0.9899 | **0.8628** | 0.8544 | RT-DETR (+0.84%) |
| **-** | **TRUNG BÌNH TOÀN BỘ (OVERALL)** | **0.9770** | **0.9752** | **0.9690** | **0.9744** | **0.9825** | **0.9679** | **0.8746** | **0.8620** | **RT-DETR nhỉnh hơn mAP** |

---

## 4. PHÂN TÍCH CHUYÊN SÂU NGUYÊN LÝ KHOA HỌC

### 4.1. Thế mạnh của RF-DETR: Tốc độ hội tụ và khả năng nắm bắt ngữ cảnh toàn cục (DINOv2)
- **Hội tụ cực nhanh (Fast Convergence)**: Nhờ cơ chế **Group DETR** (13 training groups) kết hợp trọng số tiền huấn luyện tự giám sát mạnh mẽ của **DINOv2**, RF-DETR đạt đỉnh hội tụ ngay tại **Epoch 13** và dừng sớm tại **Epoch 23**. Điều này giúp tiết kiệm tới **66.7% thời gian và chi phí tính toán GPU** so với quá trình huấn luyện 69 epochs của RT-DETR.
- **Độ thu hồi vượt trội (Recall = 97.44% vs 96.90%)**: RF-DETR phát hiện sót rất ít đối tượng. Đặc biệt, mô hình đạt độ chính xác tuyệt đối ($100\%$ mAP@50) trên 7 lớp thực phẩm: `bittergourd`, `chayote`, `eggplant`, `pork`, `shrimp`, `tofu`, và gần như tuyệt đối trên `garlic` ($0.9999$).
- **Độ chính xác vị trí mAP@50-95 vượt trội trên các lớp rau củ đặc thù**: RF-DETR vượt qua RT-DETR trên `chayote` ($+1.93\%$), `eggplant` ($+1.93\%$), `bittergourd` ($+1.80\%$), `okra` ($+1.68\%$), `pork` ($+0.99\%$), `scallion` ($+0.75\%$), `garlic` ($+0.73\%$).

### 4.2. Thế mạnh của RT-DETR: Độ chính xác tổng thể mAP và độ trễ suy luận phần cứng
- **Kiến trúc tối ưu phần cứng HGNetv2**: Nhờ sự kết hợp giữa khối trích xuất đặc trưng tích chập tối ưu phần cứng HGNetv2 và Hybrid Encoder, RT-DETR đạt tốc độ xử lý nhanh hơn (**7.50 ms / 133.3 FPS** so với **8.20 ms / 121.9 FPS** của RF-DETR).
- **Hàm mất mát phân phối hộp bao (Distribution Focal Loss - DFL)**: RT-DETR mô hình hóa tọa độ hộp bao dưới dạng phân phối xác suất liên tục, giúp định vị cực kỳ chuẩn xác các vật thể có cấu trúc giải phẫu phức tạp như gia cầm (`chicken`: $+4.27\%$, `chickenleg`: $+5.16\%$, `chickenwin`: $+4.93\%$) và củ quả có viền bầu tròn (`bottlegourd`: $+3.53\%$, `radish`: $+4.22\%$, `broccoli`: $+3.52\%$).
- **Độ chính xác tổng thể dẫn đầu**: RT-DETR đạt **mAP@50 = 98.25%** (+1.46%) và **mAP@50-95 = 87.46%** (+1.26%) nhờ được tối ưu sâu qua 58 epochs hiệu quả.

---

## 5. TRỰC QUAN HÓA MA TRẬN NHẦM LẪN (CONFUSION MATRIX)

Cả hai mô hình đều được xuất 2 phiên bản Ma trận nhầm lẫn (Counts và Normalized %) được tạo tại ngưỡng hoạt động `conf = 0.25`, hiển thị số liệu rõ ràng trên từng ô (`annot=True`):

### 5.1. RT-DETR Confusion Matrix
- **Ma trận chuẩn hóa (Normalized %)**: [rtdetr_confusion_matrix_normalized.png](file:///d:/NCKH/Train_models/reports_v4/figures/rtdetr_confusion_matrix_normalized.png)
- **Ma trận đếm số lượng (Counts)**: [rtdetr_confusion_matrix.png](file:///d:/NCKH/Train_models/reports_v4/figures/rtdetr_confusion_matrix.png)

![RT-DETR Normalized Confusion Matrix](file:///d:/NCKH/Train_models/reports_v4/figures/rtdetr_confusion_matrix_normalized.png)

### 5.2. RF-DETR Confusion Matrix
- **Ma trận chuẩn hóa (Normalized %)**: [rfdetr_confusion_matrix_normalized.png](file:///d:/NCKH/Train_models/reports_v4/figures/rfdetr_confusion_matrix_normalized.png)
- **Ma trận đếm số lượng (Counts)**: [rfdetr_confusion_matrix.png](file:///d:/NCKH/Train_models/reports_v4/figures/rfdetr_confusion_matrix.png)

![RF-DETR Normalized Confusion Matrix](file:///d:/NCKH/Train_models/reports_v4/figures/rfdetr_confusion_matrix_normalized.png)

**Nhận xét ma trận đối sánh:**
- **Đường chéo chính (True Positives)**: Cả hai mô hình đều đạt tỷ lệ tập trung cao từ **0.90 đến 1.00** dọc theo đường chéo chính cho đại đa số 32 lớp. Các con số được in trực tiếp trên từng ô giúp kiểm chứng trực quan, không còn hiện tượng ô trắng trơn không số như phiên bản mặc định của Ultralytics.
- **Tỷ lệ phân loại nhầm giữa các lớp**: Sự nhầm lẫn giữa các lớp thực phẩm hầu như bị triệt tiêu ở ngưỡng `conf = 0.25`. Nhầm lẫn nhỏ xuất hiện giữa `chicken` và nhãn nền do lát thịt mỏng hòa lẫn với nền đĩa.
- **Khả năng triệt tiêu False Positives**: Cả hai kiến trúc Transformer đều thể hiện khả năng lọc nhiễu nền xuất sắc nhờ cơ chế Attention loại bỏ các bounding box trùng lặp mà không cần dùng NMS truyền thống.

---

## 6. KHUYẾN NGHỊ TRIỂN KHAI THỰC TẾ

| Nhu cầu ứng dụng thực tế | Mô hình đề xuất | Lý do kỹ thuật thực nghiệm |
| :--- | :---: | :--- |
| **Thiết bị biên Edge AI / Camera thời gian thực (NVIDIA Jetson, Mobile)** | **RT-DETR** | Độ trễ cực thấp (**7.50 ms**), thông lượng cao (**133.3 FPS**), tương thích tối đa với bộ tăng tốc TensorRT / ONNX. |
| **Hệ thống Server Cloud / Căn-tin thông minh / Giám sát dinh dưỡng tự động** | **RT-DETR & RF-DETR (Ensemble)** | RT-DETR dẫn đầu về mAP tổng thể (98.25% mAP50), trong khi RF-DETR đạt Recall cao hơn (97.44%) và bù đắp tốt cho các loại rau củ phức tạp. |
| **Môi trường liên tục cập nhật món ăn mới (Continuous Active Learning)** | **RF-DETR** | Tốc độ hội tụ nhanh gấp 3 lần (chỉ cần **13 - 23 epochs**), tiết kiệm chi phí thuê hạ tầng GPU điện toán đám mây. |
| **Phát hiện các loại rau củ dạng lát cắt / biến dạng** | **RF-DETR** | Nhận diện vượt trội trên `bittergourd` (Khổ qua), `chayote` (Su su), `eggplant` (Cà tím), `okra` (Đậu bắp). |
| **Phát hiện nhóm thịt gia cầm / xương** | **RT-DETR** | Nhận diện vượt trội trên `chicken` (Thịt gà), `chickenleg` (Đùi gà), `chickenwin` (Cánh gà). |
