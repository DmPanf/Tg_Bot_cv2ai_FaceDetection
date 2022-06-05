# -*- coding: UTF-8 -*-
# Модуль для тестирования скорости работы и сравнения алгоритмов: HAAR, HOG, CNN

import numpy as np
import io
import cv2      # базовая библиотека Open CV2
import time     # библиотека для оценки временных характеристик
#from mtcnn import MTCNN

#cascadePath = 'tgbot/service/Cascades/haarcascade_frontalface_default.xml' # локальное размещение Модели HAAR

def mtcnn_method(in_img, out_img):

#detector = MTCNN()
#img = cv2.imread("img.jpg")
#detections = detector.detect_faces(img)
#for detection in detections:
#   score = detection["confidence"]
#   if score &amp;amp;gt; 0.90:
#      x, y, w, h = detection["box"]
#      detected_face = img[int(y):int(y+h), int(x):int(x+w)]

    mtcnn_face_detector = MTCNN()
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #image = cv2.imread(in_img) # чтение первоначального изображения

    print("Execution Time (in seconds) :")
    h, w = image.shape[:2]
    Y = h
    str_shape = ' => [' + str(w) + ' x ' + str(h) + ' px]'
    print(str_shape)

    # ---> MTCNN <---
    start = time.time()
    frame = cv2.resize(image, (600, 400))
    faces_mtcnn = mtcnn_face_detector.detect_faces(frame)
    N = str(len(faces_mtcnn))
    end = time.time()

    t_mtcnn = format(end - start, '.2f')
    str_mtcnn = 'MTCNN{' + N + ': ' + str(t_mtcnn) + ' sec'
    print(str_mtcnn)

    if boxes:
        box = boxes[0]['box']
        conf = boxes[0]['confidence']
        x, y, w, h = box[0], box[1], box[2], box[3]

        if conf > 0.5:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)

    # Вывод надписи со всеми временными характеристиками в левый верхний угол # 15 40 60 80 100 125
#    image[Y-125:Y-4, 3:223] = (195,195,127) # заливка области под надписи: координаты [y:y, x:x]
    image[Y-125:Y-4, 3:230] = (195,195,127) # заливка области под надписи: координаты [y:y, x:x]
    cv2.putText(image, str_shape, (5,Y-110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 2)
    cv2.putText(image, str_mtcnn, (5,Y-90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

    is_success, buffer = cv2.imencode(".jpg", image)
    return io.BytesIO(buffer)

#    return in_img
