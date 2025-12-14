import cv2
import os

def create_directory(directory):
    """
   Tạo thư mục nếu nó chưa tồn tại.
    Tham số:
        directory (str): Đường dẫn của thư mục cần tạo.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    # Khởi động webcam (số 0 thường là webcam mặc định)
    video = cv2.VideoCapture(0)
    # Load bộ phát hiện khuôn mặt Haar Cascade (nhẹ, nhanh, tích hợp sẵn trong OpenCV)
    facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    count = 0
    # Nhập tên người dùng để đặt tên cho thư mục dữ liệu
    nameID = str(input("Enter Your Name: ")).lower()
    # Đường dẫn lưu ảnh raw (ảnh thô chưa qua xử lý căn chỉnh)
    path = '../Dataset/FaceData/raw/' + nameID  # Đường dẫn đúng đến thư mục raw

    create_directory(path)

    while True:
        # Đọc từng khung hình từ webcam
        ret, frame = video.read()
        # Phát hiện khuôn mặt trong khung hình
        # 1.3 là scaleFactor, 5 là minNeighbors
        faces = facedetect.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in faces:
            count = count + 1
            # Tạo đường dẫn file ảnh: User-Ten-SoThuTu.jpg
            image_path = f"{path}/User-{nameID}-{count}.jpg"
            print("Creating Images........." + image_path)
            # Lưu phần ảnh chứa khuôn mặt (crop theo tọa độ x, y, w, h)
            cv2.imwrite(image_path, frame[y:y + h, x:x + w])
            # Vẽ khung chữ nhật màu xanh lá quanh mặt để hiển thị lên màn hình
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Hiển thị cửa sổ camera
        cv2.imshow("WindowFrame", frame)
        # Chờ 1ms, kiểm tra phím bấm
        cv2.waitKey(1)
        # Nếu đã chụp đủ hơn 100 ảnh thì dừng lại
        if count > 100:
            break
    # Giải phóng camera và đóng cửa sổ
    video.release()
    cv2.destroyAllWindows()
