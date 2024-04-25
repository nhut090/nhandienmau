import cv2
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# Tắt cảnh báo với UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# Đọc dữ liệu từ tệp CSV
index = ["color", "color_name", "hex", "R", "G", "B"]
df = pd.read_csv("colors.csv", names=index, header=None)

# Chia dữ liệu thành features (giá trị BGR) và labels (tên màu)
X = df[['R', 'G', 'B']].copy()
X.columns = ['R', 'G', 'B']
y = df['color_name']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Đánh giá mô hình trên tập kiểm tra
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Sử dụng mô hình để nhận diện màu sắc từ hình ảnh
def detect_color(image):
    b, g, r = cv2.split(image)
    b = b.mean()
    g = g.mean()
    r = r.mean()
    color = model.predict([[r, g, b]])
    rgb_value = (int(r), int(g), int(b))
    return color[0], rgb_value

# Vòng lặp chính để đọc hình ảnh từ camera, xử lý và hiển thị
cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định (0)
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    x, y = int(img.shape[1]/2), int(img.shape[0]/2)
    color, rgb_value = detect_color(img)
    cv2.putText(img, color, (x - 140, y - 190), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    # Hiển thị giá trị màu RGB
    rgb_text = f"RGB: {rgb_value}"
    cv2.putText(img, rgb_text, (x - 140, y - 160), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    # Vẽ một hình vuông
    square_size = 300  # Kích thước hình vuông lớn hơn
    square_color = (0, 255, 0)  # Màu xanh lá cây
    square_top_left = (x - square_size // 2, y - square_size // 2)
    square_bottom_right = (x + square_size // 2, y + square_size // 2)
    cv2.rectangle(img, square_top_left, square_bottom_right, square_color, 2)

    cv2.imshow('Color Detector', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
