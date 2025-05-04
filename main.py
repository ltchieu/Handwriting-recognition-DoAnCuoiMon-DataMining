import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import torch
import torchvision
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import cv2
import os
from PIL import Image, ImageTk
import joblib
from torchvision import transforms
from scipy.ndimage import rotate, shift
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

CANVAS_SIZE = 280

# Tắt cảnh báo joblib
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

class DigitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Main container
        main_container = tk.Frame(root, bg="#f0f0f0")
        main_container.pack(padx=20, pady=20, fill="both", expand=True)

        # Left panel - Drawing area
        left_panel = tk.Frame(main_container, bg="#f0f0f0")
        left_panel.pack(side="left", padx=10, fill="both", expand=True)

        # Canvas title
        canvas_title = tk.Label(left_panel, text="Draw a digit here", 
                            font=("Helvetica", 14, "bold"),
                            bg="#f0f0f0", fg="#333333")
        canvas_title.pack(pady=5)

        # Canvas with border
        canvas_frame = tk.Frame(left_panel, bg="#333333", padx=2, pady=2)
        canvas_frame.pack()
        self.canvas = tk.Canvas(canvas_frame, width=280, height=280, 
                            bg="white", cursor="pencil")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        # Button frame
        btn_frame = tk.Frame(left_panel, bg="#f0f0f0")
        btn_frame.pack(pady=20)

        # Styled buttons
        button_style = {
            "font": ("Helvetica", 11),
            "width": 12,
            "height": 2,
            "border": 0,
            "borderwidth": 0,
            "cursor": "hand2"
        }

        self.predict_btn = tk.Button(btn_frame, text="Predict",
                                bg="#4CAF50", fg="white",
                                command=self.predict,
                                **button_style)
        self.predict_btn.pack(side="left", padx=5)

        self.import_btn = tk.Button(btn_frame, text="Import Image",
                                bg="#2196F3", fg="white",
                                command=self.import_image,
                                **button_style)
        self.import_btn.pack(side="left", padx=5)

        self.clear_btn = tk.Button(btn_frame, text="Clear",
                                bg="#f44336", fg="white",
                                command=self.clear_canvas,
                                **button_style)
        self.clear_btn.pack(side="left", padx=5)

        # Right panel - Results
        right_panel = tk.Frame(main_container, bg="#f0f0f0")
        right_panel.pack(side="right", padx=10, fill="both")

        # Results section
        results_frame = tk.LabelFrame(right_panel, text="Results",
                                    font=("Helvetica", 12, "bold"),
                                    bg="#f0f0f0", fg="#333333",
                                    padx=10, pady=10)
        results_frame.pack(fill="both", expand=True)

        self.result_label = tk.Label(results_frame, text="Result: ",
                                font=("Helvetica", 24, "bold"),
                                bg="#f0f0f0", fg="#333333")
        self.result_label.pack(pady=20)

        # Model info section
        model_frame = tk.LabelFrame(right_panel, text="Model Information",
                                font=("Helvetica", 12, "bold"),
                                bg="#f0f0f0", fg="#333333",
                                padx=10, pady=10)
        model_frame.pack(fill="both", expand=True)

        self.model_info_label = tk.Label(model_frame, 
                                    text="Model Info: Training...",
                                    font=("Helvetica", 10),
                                    bg="#f0f0f0", fg="#666666",
                                    wraplength=250)
        self.model_info_label.pack(pady=10)

        # Probabilities section
        prob_frame = tk.LabelFrame(right_panel, text="Prediction Probabilities",
                                font=("Helvetica", 12, "bold"),
                                bg="#f0f0f0", fg="#333333",
                                padx=10, pady=10)
        prob_frame.pack(fill="both", expand=True)

        self.prob_label = tk.Label(prob_frame, text="Probabilities: ",
                                font=("Helvetica", 10),
                                bg="#f0f0f0", fg="#666666",
                                wraplength=250)
        self.prob_label.pack(pady=10)

        # Khởi tạo mảng để lưu hình ảnh
        self.image = np.ones((280, 280), dtype=np.uint8) * 255
        self.photo = None

        # Tải hoặc huấn luyện mô hình
        self.model, self.best_k, self.best_accuracy, self.pca, self.accuracies, self.k_vals = self.load_or_train_model()
        self.model_info_label.config(
            text=f"Model: K-Nearest Neighbors\nk={self.best_k}\nAccuracy={self.best_accuracy*100:.2f}%")
        # Vẽ biểu đồ sau khi huấn luyện
        self.plot_results()

    def augment_data(self, image, label, num_augmentations=3):
        """Tăng cường dữ liệu bằng cách xoay, dịch chuyển"""
        augmented_images = [image]
        augmented_labels = [label]

        for _ in range(num_augmentations):
            # Xoay ngẫu nhiên từ -10 đến 10 độ
            angle = np.random.uniform(-10, 10)
            rotated_img = rotate(image.reshape(28, 28), angle, reshape=False).reshape(-1)
            augmented_images.append(rotated_img)
            augmented_labels.append(label)

            # Dịch chuyển ngẫu nhiên
            shift_x, shift_y = np.random.randint(-2, 3, size=2)
            shifted_img = shift(image.reshape(28, 28), [shift_x, shift_y], mode='nearest').reshape(-1)
            augmented_images.append(shifted_img)
            augmented_labels.append(label)

        return np.array(augmented_images), np.array(augmented_labels)

    def load_or_train_model(self):
        # Kiểm tra xem mô hình đã được lưu chưa
        if os.path.exists('knn_model.pkl') and os.path.exists('pca_model.pkl'):
            print("Loading saved model...")
            model = joblib.load('knn_model.pkl')
            pca = joblib.load('pca_model.pkl')
            best_k = model.n_neighbors
            best_accuracy = joblib.load('best_accuracy.pkl')
            # Tải accuracies và k_vals nếu có
            if os.path.exists('accuracies.pkl') and os.path.exists('k_vals.pkl'):
                accuracies = joblib.load('accuracies.pkl')
                k_vals = joblib.load('k_vals.pkl')
            else:
                accuracies, k_vals = [], range(1, 16, 2)  # Giá trị mặc định nếu không có file
            return model, best_k, best_accuracy, pca, accuracies, k_vals

        # Tải MNIST từ torchvision
        DOWNLOAD_MNIST = not (os.path.exists('./mnist/') and os.listdir('./mnist/'))
        train_data = torchvision.datasets.MNIST(
            root='./mnist/',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=DOWNLOAD_MNIST,
        )
        test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

        # Chuẩn bị dữ liệu huấn luyện
        train_x = torch.unsqueeze(train_data.data, dim=1).type(torch.FloatTensor)/255.0
        train_y = train_data.targets
        train_x = train_x.view(-1, 28*28).numpy()

        # Tăng cường dữ liệu
        print("Augmenting data...")
        augmented_x, augmented_y = [], []
        for i in range(len(train_x)):
            aug_x, aug_y = self.augment_data(train_x[i], train_y[i], num_augmentations=2)
            augmented_x.append(aug_x)
            augmented_y.append(aug_y)
        train_x = np.vstack(augmented_x)
        train_y = np.hstack(augmented_y)
        print(f"Augmented training data: {train_x.shape[0]} samples")

        # Chuẩn bị dữ liệu kiểm tra
        test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)/255.0
        test_y = test_data.targets
        test_x = test_x.view(-1, 28*28).numpy()

        # Giảm chiều bằng PCA
        pca = PCA(n_components=700)
        train_x = pca.fit_transform(train_x)
        test_x = pca.transform(test_x)

        # Tìm giá trị k tối ưu
        k_vals = range(1, 16, 2)
        accuracies = []
        for k in k_vals:
            model = KNeighborsClassifier(n_neighbors=k, weights='distance')
            model.fit(train_x, train_y)
            score = model.score(test_x, test_y)
            accuracies.append(score)
            print(f"k={k}, accuracy={score*100:.2f}%")

        # Chọn k tốt nhất
        best_k = k_vals[np.argmax(accuracies)]
        best_accuracy = max(accuracies)

        # Huấn luyện mô hình cuối cùng với k tốt nhất
        model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
        model.fit(train_x, train_y)

        # Đánh giá trên tập kiểm tra
        predictions = model.predict(test_x)
        print("EVALUATION ON TESTING DATA")
        print(classification_report(test_y, predictions))

        # Lưu mô hình và accuracies
        joblib.dump(model, 'knn_model.pkl')
        joblib.dump(pca, 'pca_model.pkl')
        joblib.dump(best_accuracy, 'best_accuracy.pkl')
        joblib.dump(accuracies, 'accuracies.pkl')
        joblib.dump(list(k_vals), 'k_vals.pkl')

        return model, best_k, best_accuracy, pca, accuracies, k_vals

    def plot_results(self):
        """Vẽ và lưu các biểu đồ sau khi huấn luyện mô hình"""
        
        plot_files = [
            'best_accuracy_plot.png',
            'knn_model_plot.png',
            'pca_model_plot.png',
            'confusion_matrix_plot.png',
            'model_evaluation_plot.png'
        ]
        
        # Kiểm tra xem tất cả các biểu đồ đã tồn tại chưa
        all_plots_exist = all(os.path.exists(plot_file) for plot_file in plot_files)
        
        if all_plots_exist:
            print("All plots already exist, skipping plot generation.")
            return
        
        # Dữ liệu đã có sẵn từ self.model, self.best_accuracy, self.pca
        # --- Plot 1: Best Accuracy ---
        plt.figure(figsize=(6, 4))
        plt.bar(['Best Accuracy'], [self.best_accuracy], color='blue')
        plt.title('Best Model Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.savefig('best_accuracy_plot.png')
        print('Lưu xong plot1')
        plt.close()

        # --- Plot 2: KNN Model (Accuracy vs. k) ---
        plt.figure(figsize=(8, 5))
        plt.plot(self.k_vals, self.accuracies, marker='o', color='green')
        plt.title('KNN Model: Accuracy vs. k')
        plt.xlabel('k (Number of Neighbors)')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig('knn_model_plot.png')
        print('Lưu xong plot2')
        plt.close()

        # --- Plot 3: PCA Model (Cumulative Explained Variance Ratio) ---
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='red')
        plt.title('PCA Model: Cumulative Explained Variance Ratio')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Variance Ratio')
        plt.grid(True)
        plt.savefig('pca_model_plot.png')
        print('Lưu xong plot3')        
        plt.close()
        
        # --- Plot 4: Confusion Matrix (Thay thế Phân bố nhãn) ---
        test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
        test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor) / 255.0
        test_y = test_data.targets.numpy()
        test_x = test_x.view(-1, 28 * 28).numpy()
        test_x = self.pca.transform(test_x)
        predictions = self.model.predict(test_x)
        cm = confusion_matrix(test_y, predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix on Test Data')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix_plot.png')
        print('Lưu xong plot4')
        plt.close()
        
        # --- Plot 5: Model Evaluation (Thay thế Phân bố khoảng cách láng giềng) ---
        # Tính các chỉ số từ classification_report
        report = classification_report(test_y, predictions, output_dict=True)
        metrics = {
            'Accuracy': report['accuracy'],
            'Macro Precision': report['macro avg']['precision'],
            'Macro Recall': report['macro avg']['recall'],
            'Macro F1-Score': report['macro avg']['f1-score']
        }

        plt.figure(figsize=(8, 5))
        plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'red', 'purple'])
        plt.title('Model Evaluation Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.savefig('model_evaluation_plot.png')
        print('Lưu xong plot5')
        plt.close()                                
        
        print("Plots saved as 'best_accuracy_plot.png', 'knn_model_plot.png', 'pca_model_plot.png', 'confusion_matrix_plot.png', 'model_evaluation_plot.png'")
        
    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        cv2.circle(self.image, (x, y), r, 0, -1)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = np.ones((280, 280), dtype=np.uint8) * 255
        self.result_label.config(text="Result: ")
        self.prob_label.config(text="Probabilities: ")
        self.photo = None

    def display_image_on_canvas(self, img):
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((200, 200), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(100, 100, image=self.photo)

    def preprocess_image(self, img):
        # Làm mịn để giảm nhiễu
        img = cv2.GaussianBlur(img, (5, 5), 0)
        # Ngưỡng hóa để làm rõ ranh giới
        _, img_thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Làm dày nét
        kernel = np.ones((3, 3), np.uint8)
        img_thresh = cv2.dilate(img_thresh, kernel, iterations=1)
        # Thay đổi kích thước về 28x28
        img_resized = cv2.resize(img_thresh, (28, 28), interpolation=cv2.INTER_AREA)
        # Chuẩn hóa và đảo màu
        img_array = img_resized / 255.0
        img_array = 1 - img_array
        img_flat = img_array.flatten().reshape(1, -1)
        # Áp dụng PCA
        img_flat = self.pca.transform(img_flat)
        return img_flat, img_thresh

    def predict_with_prob(self, img_flat):
        prediction = self.model.predict(img_flat)[0]
        probs = self.model.predict_proba(img_flat)[0]
        prob_text = ", ".join([f"{i}: {prob*100:.1f}%" for i, prob in enumerate(probs)])
        return prediction, prob_text, probs

    def import_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if not file_path:
            return

        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                messagebox.showerror("Error", "Cannot load image!")
                return

            img_flat, img_processed = self.preprocess_image(img)
            prediction, prob_text = self.predict_with_prob(img_flat)

            self.result_label.config(text=f"Result: {prediction}")
            self.prob_label.config(text=f"Probabilities: {prob_text}")

            self.image = cv2.resize(img_processed, (200, 200), interpolation=cv2.INTER_AREA)
            self.display_image_on_canvas(self.image)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def predict(self):
        img_flat, img_processed = self.preprocess_image(self.image)
        prediction, prob_text, probs = self.predict_with_prob(img_flat)
        confidence = np.max(probs)
        
        
        self.result_label.config(text=f"Result: {prediction}")
        self.prob_label.config(text=f"Probabilities: {prob_text}")
        
        if(confidence < 0.77):
            self.result_label.config(text="Kết quả: Không phải chữ số", fg="red", font=("Helvetica", 16, "bold"))
        else:
            self.image = cv2.resize(img_processed, (200, 200), interpolation=cv2.INTER_AREA)
            self.display_image_on_canvas(self.image)

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitClassifierApp(root)
    root.mainloop()