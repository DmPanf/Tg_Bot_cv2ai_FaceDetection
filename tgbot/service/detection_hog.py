# -*- coding: UTF-8 -*-
# Модуль для тестирования скорости работы и сравнения алгоритмов: HAAR, HOG, CNN

import numpy as np
import io
import cv2      # базовая библиотека Open CV2
import time     # библиотека для оценки временных характеристик
import dlib     # необходимая библиотека для алгоритмов HOG и CNN

#cascadePath = 'tgbot/service/Cascades/haarcascade_frontalface_default.xml' # локальное размещение Модели HAAR

def hog_method(in_img, out_img):
    hog_face_detector = dlib.get_frontal_face_detector()               # HOG + SVM детектор лица
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #image = cv2.imread(in_img) # чтение первоначального изображения

    print("Execution Time (in seconds) :")
    h, w = image.shape[:2]
    Y = h
    str_shape = ' => [' + str(w) + ' x ' + str(h) + ' px]'
    print(str_shape)

    # ---> HOG <---
    start = time.time()
    faces_hog = hog_face_detector(image, 1) # HOG
    N = str(len(faces_hog))
    end = time.time()

    t_hog = format(end - start, '.2f')
    str_hog = ' HOG{' + N + ': ' + str(t_hog) + ' sec'
    print(str_hog)

    for face in faces_hog:  # перебор (для случая обнаружения нескольких ROI) HOG
        x3 = face.left()
        y3 = face.top()
        w3 = face.right() - x3
        h3 = face.bottom() - y3
        cv2.rectangle(image, (x3,y3), (x3+w3,y3+h3), (0,255,0), 2)    # отрисовка рамки  ROI

    # Вывод надписи со всеми временными характеристиками в левый верхний угол # 15 40 60 80 100 125
#    image[Y-125:Y-4, 3:223] = (195,195,127) # заливка области под надписи: координаты [y:y, x:x]
    image[Y-125:Y-4, 3:230] = (195,195,127) # заливка области под надписи: координаты [y:y, x:x]
    cv2.putText(image, str_shape, (5,Y-110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 2)
    cv2.putText(image, str_hog, (5,Y-90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    is_success, buffer = cv2.imencode(".jpg", image)
    return io.BytesIO(buffer)



#    return in_img
