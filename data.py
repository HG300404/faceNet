import cv2
import os
import numpy as np
import face_recognition

def load_and_preprocess_image(image_path):
    """
    Đọc và tiền xử lý ảnh.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể tải ảnh từ {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (160, 160))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Chuẩn hóa ảnh
    return image

def extract_face(image):
    """
    Phát hiện và trích xuất khuôn mặt từ ảnh.
    """
    bboxs = face_recognition.face_locations(image)
    if len(bboxs) == 0:
        return None
    (startY, startX, endY, endX) = bboxs[0]
    face = image[startY:endY, startX:endX]
    return face

def save_face(face, person_name, count):
    """
    Lưu khuôn mặt vào thư mục dữ liệu.
    """
    directory = f"faces/{person_name}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{directory}/{person_name}_{count}.jpg"
    cv2.imwrite(filename, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

def collect_faces_from_data(data_path):
    """
    Thu thập khuôn mặt từ thư mục dữ liệu.
    """
    count = 0
    for person_name in os.listdir(data_path):
        person_path = os.path.join(data_path, person_name)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                image = load_and_preprocess_image(img_path)
                face = extract_face(image)
                if face is not None:
                    save_face(face, person_name, count)
                    count += 1
data_path = "data"
collect_faces_from_data(data_path)
