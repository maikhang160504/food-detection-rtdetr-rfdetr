Yêu cầu train cho 2 mô hình RT-DETR và RF-DETR trên modal (patience = 20), lưu trữ các phần cần thiết ở storage:
# dataset : https://app.roboflow.com/nckhcict2025/completed-project/5 
Lưu loss (tổng train loss, loss theo class, loss theo bouding, GIoU, learning rate)
Box Loss là tên gọi chung cho sai số vẽ khung của mô hình, trong khi GIoU là một hàm mất mát tiên tiến nằm bên trong Box Loss giúp giải quyết bài toán khoảng cách khi các đối tượng (nguyên liệu) nằm rời rạc hoặc chồng chéo, đảm bảo mô hình định vị khung hình chuẩn xác tuyệt đối.
Thiết lập 50 epochs
Lưu checkpoint:
+ Lưu best.pt/best.pth theo map@50-95 cao nhất trên tập Validation
+ Lưu thêm last.pt/last.pth (epoch cuối)
File báo cáo cần có:
Log theo từng epoch chứa: tổng train loss, loss thành phần: class, box regression, GIoU, learning rate và thời gian chạy thực tế => Mẫu

Epoch
Train loss
Class loss
Box loss
GIoU
Learning rate
Val Precision
Val Recall
Val mAP50
Val mAP50-95
Time(s)


Độ chính xác (precision, recall, map50, map50-95)
Độ chính xác của từng lớp (dựa trên tập test) 
Độ chính xác tổng thể trên tập dữ liệu

Tham khảo về box loss và GIoU
1. "Box Loss" (Tổng mức độ sai sót)
- Hình dung: Nó giống như điểm phạt tổng hợp mà giáo viên (hàm mất mát) chấm cho mô hình của bạn.
- Giáo viên nhìn vào khung màu đỏ của bạn và nói: "Khung này vừa bị vẽ to quá, vừa bị dịch sang bên trái 5 cm, lại còn xoay sai góc một chút. Tổng điểm phạt cho việc vẽ sai này (Box Loss) là 8.5 điểm!"
- Như vậy, Box Loss chỉ là một con số tổng kết cho biết mô hình vẽ khung đang tệ hay tốt, chứ bản thân nó chưa chỉ ra cụ thể cách sửa thế nào trong từng tình huống khó.
2. "GIoU" (Cơ chế toán học giúp kéo khung về đúng chỗ)
- Hình dung: GIoU giống như chiếc la bàn hoặc lực hút nam châm giúp mô hình biết phải kéo cái khung đỏ dịch đi đâu.
- Tình huống đặc biệt (Lý do GIoU ra đời):
- Giả sử mô hình vẽ khung đỏ lệch đi một khoảng rất xa, hoàn toàn không chạm chút nào vào củ cà rốt thực tế (giống như hai vật nằm ở hai góc bàn khác nhau).
- Nếu dùng các thuật toán Box Loss kiểu cũ (tính khoảng cách đơn thuần), lúc này khoảng cách bằng 0 hoặc không có điểm giao nhau, mô hình sẽ bị "mù", nó đứng im và không biết phải dịch khung sang trái hay sang phải vì lực kéo bằng 0.
- GIoU xuất hiện để cứu nguy: Nó tưởng tượng ra một vùng không gian bao trùm lớn nhất trùm lên cả củ cà rốt lẫn cái khung vẽ sai của bạn. Dựa vào khoảng trống của vùng bao trùm đó, GIoU tạo ra một "lực kéo" bắt buộc cái khung đỏ phải dịch chuyển lại gần củ cà rốt.
