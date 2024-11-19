import face_recognition
import os
import pickle

def train_model(data_path):
    """
    Huấn luyện mô hình nhận diện khuôn mặt.
    """
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(data_path):
        person_path = os.path.join(data_path, person_name)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                image = face_recognition.load_image_file(img_path)
                face_encoding = face_recognition.face_encodings(image)
                if face_encoding:
                    known_face_encodings.append(face_encoding[0])
                    known_face_names.append(person_name)

    # Lưu mô hình đã huấn luyện
    with open("face_model.pkl", "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)

train_model("faces")
