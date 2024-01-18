import mediapipe as mp
import cv2
import cv2
import mediapipe as mp
import string
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from functools import partial
from keras.preprocessing.image import ImageDataGenerator, array_to_img
def detect():
    model=tf.keras.models.load_model('model')
    dim = (28, 28) # 图像维度
    letters = list(string.ascii_lowercase) # 识别的字母

    x0 = 1920 // 2 - 400 # 400px left of center
    x1 = 1920 // 2 + 400 # 400px right of center
    y0 = 1080 // 2 - 400 # 400px right of center
    y1 = 1080 // 2 + 400 # 400px right of center

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)

    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        print(frame.shape)
        cropped = frame[int(0.5*frame.shape[0]):int(1.5*frame.shape[0]), int(0.5*frame.shape[1]):int(1.5*frame.shape[1])] # 截取
        img = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) # 转成灰度图
        img = cv2.GaussianBlur(img, (5, 5), 0) # 图像平滑
        img = cv2.resize(img, dim) # 图像大小缩放
        img = np.reshape(img, (1,img.shape[0],img.shape[1],1))
        img = tf.cast(img, tf.float32)
        pred=model.predict(img)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame= cv2.flip(frame,1)
        results = hands.process(frame)

        


        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame,letters[np.argmax(pred[0])],(0,100),0,1.3,(0,0,255),3)
        cv2.imshow('MediaPipe Hands', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()

if __name__ == '__main__':
    detect()
    cv2.destroyAllWindows()