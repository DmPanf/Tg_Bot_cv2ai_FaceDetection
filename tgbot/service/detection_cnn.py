# -*- coding: UTF-8 -*-
# Модуль для тестирования скорости работы и сравнения алгоритмов: HAAR, HOG, CNN

import numpy as np
import io
import cv2      # базовая библиотека Open CV2
import time     # библиотека для оценки временных характеристик
import dlib     # необходимая библиотека для алгоритмов HOG и CNN

#cascadePath = 'tgbot/service/Cascades/haarcascade_frontalface_default.xml' # локальное размещение Модели HAAR
weights = 'tgbot/service/Cascades/mmod_human_face_detector.dat' # CNN

def cnn_method(in_img, out_img):
    cnn_face_detector = dlib.cnn_face_detection_model_v1(weights)      # CNN детектор лица
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #image = cv2.imread(in_img) # чтение первоначального изображения

    print("Execution Time (in seconds) :")
    h, w = image.shape[:2]
    Y = h
    str_shape = ' => [' + str(w) + ' x ' + str(h) + ' px]'
    print(str_shape)

    # ---> CNN <---
    start = time.time()
    faces_cnn = cnn_face_detector(image, 1) # CNN
    N = str(len(faces_cnn))
    end = time.time()

    conf = 0.0
    for face in faces_cnn:   # перебор (для случая обнаружения нескольких ROI) CNN
        x4 = face.rect.left()
        y4 = face.rect.top()
        w4 = face.rect.right() - x4
        h4 = face.rect.bottom() - y4
        conf = face.confidence # достоверность
        cv2.rectangle(image, (x4,y4), (x4+w4,y4+h4), (0,0,255), 2)  # отрисовка рамки  ROI

    if conf > 0:
        prob = format(conf * 100, '.2f')
    else:
        prob = 0.0

    t_cnn = format(end - start, '.2f')
    str_cnn = ' CNN{' + N + '}: ' + str(t_cnn) + ' sec [' + str(prob) + ']'
    print(str_cnn)

    # Вывод надписи со всеми временными характеристиками в левый верхний угол # 15 40 60 80 100 125
#    image[Y-125:Y-4, 3:223] = (195,195,127) # заливка области под надписи: координаты [y:y, x:x]
    image[Y-125:Y-4, 3:230] = (195,195,127) # заливка области под надписи: координаты [y:y, x:x]
    cv2.putText(image, str_shape, (5,Y-110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 2)
    cv2.putText(image, str_cnn, (5,Y-90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    is_success, buffer = cv2.imencode(".jpg", image)
    return io.BytesIO(buffer)

#    return in_img
