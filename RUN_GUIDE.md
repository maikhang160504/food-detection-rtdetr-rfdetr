# HƯỚNG DẪN CHẠY PIPELINE HUẤN LUYỆN RT-DETR & RF-DETR TRÊN MODAL GPU

Tài liệu này hướng dẫn chi tiết từng bước để chuẩn bị, cấu hình, chạy huấn luyện 50 epochs, theo dõi tiến độ và tải kết quả/checkpoint về máy tính cá nhân.

---

## 1. Yêu Cầu Chuẩn Bị Trước Khi Chạy

### 1.1. Cài đặt thư viện trên máy Local
Mở PowerShell/Terminal tại thư mục dự án `Train_models` và chạy:
```powershell
pip install -r requirements.txt
```

### 1.2. Đăng nhập Modal CLI
Nếu chưa đăng nhập Modal:
```powershell
modal setup
```
*(Trình duyệt sẽ mở ra để bạn xác thực tài khoản Modal. Sau khi đăng nhập, terminal sẽ báo thành công).*

### 1.3. Lấy Roboflow API Key
1. Truy cập [Roboflow App](https://app.roboflow.com/).
2. Đăng nhập vào workspace `nckhcict2025`.
3. Vào **Settings** $\rightarrow$ **Roboflow API** $\rightarrow$ Sao chép **Private API Key**.

---

## 2. Trạng Thái Secret Trên Modal

Secret `roboflow-secret` (chứa `ROBOFLOW_API_KEY`) **đã được tạo thành công trên Modal**. Các container khi khởi chạy sẽ tự động nạp key từ secret này mà không cần nhập thủ công.

---

## 3. Quy Trình Chạy Chuẩn (2 Bước Nhanh & Tối Ưu Nhất)

### BƯỚC 1: Tải và Kiểm Tra Dataset Vào Modal Volume (Chỉ Chạy 1 Lần)
Lệnh này sẽ tải 16k ảnh từ Roboflow và lưu vĩnh viễn vào Modal Volume `/vol/datasets/`:
```powershell
modal run modal_app/pipeline.py::download_dataset_step
```
*(Sau khi bước 1 chạy xong, dataset đã nằm chắc chắn trong Volume. Bạn không cần tải lại nữa).*

---

### BƯỚC 2: Huấn Luyện SONG SONG 2 Mô Hình Trên 2 GPU NVIDIA A100
Khi dataset đã có trong Volume, bạn chỉ cần kích hoạt lệnh sau để chạy **đồng thời cả 2 mô hình trên 2 GPU A100 song song**:
```powershell
modal run modal_app/pipeline.py::train_parallel
```
- **1 GPU A100** chạy RT-DETR (50 epochs).
- **1 GPU A100** chạy RF-DETR (50 epochs) song song cùng lúc.
- Khi cả 2 train xong $\rightarrow$ Hệ thống tự động chuyển sang bước **Đánh giá trên tập Test** và xuất báo cáo so sánh.
- **Tổng thời gian**: Chỉ mất khoảng **~1 giờ 15 phút** cho toàn bộ quá trình!

---

### Cách Khác: Chạy Từng Model Riêng Lẻ (Nếu Muốn)

- **Chỉ train RT-DETR (1 GPU A100)**:
  ```powershell
  modal run modal_app/pipeline.py::train_rtdetr_step
  ```
- **Chỉ train RF-DETR (1 GPU A100)**:
  ```powershell
  modal run modal_app/pipeline.py::train_rfdetr_step
  ```
- **Chỉ chạy Đánh giá & Xuất báo cáo so sánh**:
  ```powershell
  modal run modal_app/pipeline.py::evaluate_both_models
  ```

---

## 4. Theo Dõi Tiến Độ Trực Tiếp (Monitoring)

1. **Xem trực tiếp trên Terminal**:
   - Khi chạy lệnh `modal run`, log của từng epoch sẽ in trực tiếp ra màn hình terminal kèm theo các chỉ số: `Epoch`, `Train loss`, `Class loss`, `Box loss`, `GIoU`, `Learning rate`, `Epoch Time (s)`.
2. **Xem trên Web Dashboard của Modal**:
   - Mở đường link `https://modal.com/apps` hiển thị trên terminal để xem biểu đồ sử dụng GPU (GPU Utilization, VRAM Memory, GPU Temperature, Logs).

---

## 5. Cơ Chế Tự Động Resume Khi Bị Ngắt Ngang / Hết Giờ

- Nếu job bị dừng đột ngột (do mất mạng, hết hạn mức thời gian, hoặc bạn chủ động bấm `Ctrl+C`):
  - Checkpoint gần nhất (`last.pt` / `last.pth`) và file log CSV đều **đã được lưu và commit an toàn trong Modal Volume**.
- **Cách tiếp tục**:
  - Bạn chỉ cần gõ lại lệnh huấn luyện tương ứng (ví dụ `modal run modal_app/pipeline.py::train_rtdetr_step`), hệ thống sẽ **tự động phát hiện checkpoint dở dang và tiếp tục train tiếp từ epoch đó**, không bị train lại từ đầu và không bị mất log các epoch trước.

---

## 6. Chương Trình Đồng Bộ Tự Động Từ Modal Storage Về Laptop

Chúng tôi đã xây dựng sẵn 2 công cụ giúp bạn đồng bộ toàn bộ Checkpoints, Logs CSV và Báo cáo đánh giá về laptop:

### Cách 1: Dùng Python Script (Đa năng & Tùy biến)
Chạy lệnh trực tiếp trong PowerShell:
```powershell
# Tải toàn bộ Checkpoints, Logs và Kết quả đánh giá về thư mục ./synced_results
python sync_storage.py

# Hoặc chỉ tải Checkpoints (best.pt, best.pth, last.pt, last.pth)
python sync_storage.py --checkpoints-only

# Hoặc chỉ tải Logs CSV và Báo cáo so sánh
python sync_storage.py --logs-only
```

---

### Cách 2: Dùng PowerShell Script 1-Click
```powershell
.\sync_storage.ps1
```

Toàn bộ dữ liệu sau khi tải sẽ nằm gọn gàng tại thư mục:
`D:\NCKH\Train_models\synced_results\`
- `checkpoints/`: Chứa `best.pt`, `last.pt`, `best.pth`, `last.pth`.
- `logs/`: Chứa bảng `rtdetr_epoch_logs.csv`, `rfdetr_epoch_logs.csv` và `comparison_report.md`.
- `outputs/`: Chứa kết quả đánh giá chi tiết trên tập Test.

---

## 7. Cấu Trúc Các File Kết Quả Đầu Ra

Sau khi hoàn tất quá trình huấn luyện và đánh giá, bạn sẽ có các file quan trọng sau:

| Đường dẫn trên Volume | Nội dung & Chức năng |
| :--- | :--- |
| `/vol/checkpoints/rtdetr/best.pt` | Checkpoint tốt nhất của RT-DETR (đạt `mAP@50-95` cao nhất trên tập Validation). |
| `/vol/checkpoints/rtdetr/last.pt` | Checkpoint epoch 50 của RT-DETR. |
| `/vol/checkpoints/rfdetr/best.pth` | Checkpoint tốt nhất của RF-DETR (đạt `mAP@50-95` cao nhất trên tập Validation). |
| `/vol/checkpoints/rfdetr/last.pth` | Checkpoint epoch 50 của RF-DETR. |
| `/vol/logs/rtdetr/rtdetr_epoch_logs.csv` | File bảng log 50 epochs của RT-DETR (gồm 6 thông số loss, GIoU, LR, runtime). |
| `/vol/logs/rfdetr/rfdetr_epoch_logs.csv` | File bảng log 50 epochs của RF-DETR (gồm 6 thông số loss, GIoU, LR, runtime). |
| `/vol/logs/comparison_report.md` | Báo cáo so sánh tổng hợp chi tiết giữa RT-DETR và RF-DETR. |
| `/vol/outputs/rtdetr_eval/` | Kết quả đánh giá Precision, Recall, mAP tổng thể và từng lớp của RT-DETR trên tập Test. |
| `/vol/outputs/rfdetr_eval/` | Kết quả đánh giá Precision, Recall, mAP tổng thể và từng lớp của RF-DETR trên tập Test. |

---

## 8. Xử Lý Các Vấn Đề Thường Gặp (Troubleshooting)

1. **Lỗi `ROBOFLOW_API_KEY is not set`**:
   - Chạy `modal secret create roboflow-secret ROBOFLOW_API_KEY=your_key` hoặc truyền cờ `--roboflow-key "your_key"` vào lệnh `modal run`.
2. **Lỗi hết dung lượng disk hoặc OOM**:
   - Cấu hình mặc định dùng GPU `A10G` (24GB VRAM) với batch size 16 (RT-DETR) và batch size 8 (RF-DETR), đảm bảo hoạt động ổn định. Nếu muốn giảm thêm bộ nhớ, bạn có thể chỉnh `batch_size` trong `configs/rtdetr.yaml` hoặc `configs/rfdetr.yaml`.
