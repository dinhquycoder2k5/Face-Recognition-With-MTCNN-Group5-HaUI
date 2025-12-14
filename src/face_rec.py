from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


import argparse
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    # Cài đặt các tham số cần thiết
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7] # Ngưỡng phát hiện mặt của MTCNN
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160 # Kích thước ảnh đầu vào cho FaceNet
    CLASSIFIER_PATH = 'Models/facemodel.pkl' # File model SVM đã train ở bước classifier.py
    VIDEO_PATH = args.path # 0 là webcam, hoặc đường dẫn file video
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb' # File model FaceNet gốc (Pre-trained)

    # Load model SVM (Classifier)
    try:
        with open(CLASSIFIER_PATH, 'rb') as file:
            model, class_names = pickle.load(file)
        print("Custom Classifier, Successfully loaded")
    except FileNotFoundError:
        print(f"Error: Khong tim thay file {CLASSIFIER_PATH}")
        return

    with tf.Graph().as_default():
        # Cài đặt GPU nếu có (giới hạn bộ nhớ 60%)
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            # Load model FaceNet để trích xuất đặc trưng
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Lấy tensor input và output từ graph
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Khởi tạo mạng MTCNN
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            people_detected = set()
            person_detected = collections.Counter()

            # Lấy hình ảnh từ file video hoặc webcam
            cap = cv2.VideoCapture(VIDEO_PATH)

            while (cap.isOpened()):
                # Đọc từng frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Phát hiện khuôn mặt bằng MTCNN, trả về vị trí trong bounding_boxes
                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    # Nếu có ít nhất 1 khuôn mặt trong frame
                    if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # Cắt phần khuôn mặt tìm được
                            # Kiểm tra biên để tránh lỗi Crash OpenCV
                            if bb[i][0] < 0 or bb[i][1] < 0 or bb[i][2] >= frame.shape[1] or bb[i][3] >= frame.shape[0]:
                                continue

                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            
                            # Resize về kích thước chuẩn input của FaceNet (160x160)
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                interpolation=cv2.INTER_CUBIC)
                            # Tiền xử lý (làm trắng/chuẩn hóa ảnh)
                            scaled = facenet.prewhiten(scaled)
                            
                            # Reshape để phù hợp với input tensor (Batch_size, Height, Width, Channel)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            
                            # Chạy FaceNet để lấy embedding (vector đặc trưng)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)

                            # Đưa vector đặc trưng vào model SVM để phân loại (đây là ai?)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1) # Index của class có xác suất cao nhất
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices] # Giá trị xác suất cao nhất

                            # Lấy ra tên
                            best_name = class_names[best_class_indices[0]]
                            print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                            # Vẽ khung màu xanh quanh khuôn mặt
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20

                            # Nếu tỷ lệ nhận dạng > 0.5 (50%) thì hiển thị tên
                            if best_class_probabilities > 0.5:
                                name = class_names[best_class_indices[0]]
                            else:
                                # Còn nếu <= 0.5 thì hiển thị Unknown (người lạ)
                                name = "Unknown"

                            # Viết tên và độ chính xác lên trên frame
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                            person_detected[best_name] += 1
                except Exception as e:
                    print(e)
                    pass

                # Hiển thị frame lên màn hình
                cv2.imshow('Face Recognition', frame)
                # Bấm 'q' để thoát
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()