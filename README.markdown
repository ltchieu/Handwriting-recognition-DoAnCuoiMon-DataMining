# Ứng dụng Nhận diện Chữ số Viết tay

## Giới thiệu
Ứng dụng này là một công cụ nhận diện chữ số viết tay (từ 0 đến 9) sử dụng thuật toán K-Nearest Neighbors (KNN) kết hợp với giảm chiều dữ liệu bằng PCA. Ứng dụng được xây dựng bằng Python và giao diện người dùng sử dụng thư viện Tkinter. Dữ liệu huấn luyện được lấy từ bộ dữ liệu MNIST, và mô hình đã được tối ưu để đạt độ chính xác cao (khoảng 97% trở lên).

## Yêu cầu
Trước khi chạy ứng dụng, bạn cần cài đặt các thư viện cần thiết. Bạn có thể cài đặt chúng bằng cách chạy lệnh sau trong terminal hoặc command prompt:

```bash
pip install -r requirements.txt
```

Các thư viện cần thiết bao gồm:
- `tkinter`
- `numpy`
- `torch` và `torchvision`
- `scikit-learn`
- `opencv-python`
- `pillow`
- `joblib`
- `matplotlib` và `seaborn`

## Chức năng chính
Ứng dụng cung cấp các chức năng sau:

1. **Vẽ và dự đoán chữ số**:
   - Người dùng có thể vẽ trực tiếp một chữ số (từ 0 đến 9) trên canvas bên trái giao diện.
   - Nhấn nút "Predict" để dự đoán chữ số. Kết quả sẽ hiển thị ở phần "Results", cùng với xác suất dự đoán cho từng chữ số (trong phần "Prediction Probabilities").

2. **Import ảnh và dự đoán**:
   - Người dùng có thể nhập một ảnh chứa chữ số viết tay từ máy tính bằng cách nhấn nút "Import Image".
   - Ứng dụng hỗ trợ các định dạng ảnh: `.png`, `.jpg`, `.jpeg`, `.bmp`.
   - Sau khi chọn ảnh, ứng dụng sẽ hiển thị ảnh đã xử lý trên canvas và dự đoán chữ số.

3. **Xóa canvas**:
   - Nhấn nút "Clear" để xóa toàn bộ nội dung trên canvas và đặt lại kết quả dự đoán.

4. **Thông tin mô hình**:
   - Phần "Model Information" hiển thị thông tin về mô hình, bao gồm giá trị k của KNN và độ chính xác trên tập kiểm tra.

5. **Biểu đồ phân tích**:
   - Sau khi huấn luyện mô hình, ứng dụng tự động tạo và lưu các biểu đồ phân tích (nếu chưa tồn tại), bao gồm:
     - `best_accuracy_plot.png`: Độ chính xác tốt nhất của mô hình.
     - `knn_model_plot.png`: Độ chính xác theo giá trị k của KNN.
     - `pca_model_plot.png`: Tỷ lệ phương sai tích lũy của PCA.
     - `confusion_matrix_plot.png`: Ma trận nhầm lẫn trên tập kiểm tra.
     - `model_evaluation_plot.png`: Các chỉ số đánh giá mô hình (Accuracy, Precision, Recall, F1-Score).

## Hướng dẫn sử dụng

### 1. Chạy ứng dụng
- Đặt file `main.py` và các file cần thiết khác vào cùng một thư mục.
- Mở terminal hoặc command prompt, di chuyển đến thư mục chứa file `main.py`, và chạy lệnh sau:

```bash
python main.py
```

- Giao diện ứng dụng sẽ xuất hiện với một canvas bên trái và các thông tin kết quả bên phải.

### 2. Vẽ và dự đoán chữ số
- Sử dụng chuột để vẽ một chữ số (từ 0 đến 9) trên canvas.
- Nhấn nút "Predict" để xem kết quả dự đoán.
- Kết quả sẽ hiển thị ở phần "Results". Nếu độ tin cậy (confidence) của dự đoán dưới 0.6, ứng dụng sẽ thông báo "Kết quả: Không phải chữ số".

### 3. Import ảnh để dự đoán
- Nhấn nút "Import Image" để chọn một ảnh từ máy tính.
- Chọn một ảnh chứa chữ số viết tay (định dạng `.png`, `.jpg`, `.jpeg`, hoặc `.bmp`).
- Sau khi chọn, ảnh sẽ được hiển thị trên canvas (sau khi xử lý), và kết quả dự đoán sẽ xuất hiện.

#### Lưu ý khi import ảnh:
- **Định dạng ảnh**: Ảnh cần là ảnh grayscale (đen trắng) và chứa một chữ số viết tay rõ ràng trên nền trắng hoặc sáng. Nếu ảnh không phải grayscale, ứng dụng có thể gặp lỗi.
- **Vị trí thư mục ảnh**: Bạn có thể đặt ảnh ở bất kỳ thư mục nào trên máy tính, miễn là đường dẫn hợp lệ. Ứng dụng sẽ mở một hộp thoại để bạn chọn file.
- **Kích thước ảnh**: Ảnh có thể có bất kỳ kích thước nào, ứng dụng sẽ tự động thay đổi kích thước về 28x28 để xử lý.
- **Chất lượng ảnh**: Đảm bảo chữ số trong ảnh rõ ràng, không bị nhòe hoặc quá nhiều nhiễu để đạt kết quả dự đoán tốt nhất.

### 4. Xóa canvas
- Nhấn nút "Clear" để xóa nội dung trên canvas và đặt lại kết quả dự đoán.