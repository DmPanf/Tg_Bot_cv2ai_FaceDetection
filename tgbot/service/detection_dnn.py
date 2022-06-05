# -*- coding: UTF-8 -*-
# Модуль для тестирования скорости работы и сравнения алгоритмов: HAAR, HOG, CNN

import numpy as np
import io
import cv2      # базовая библиотека Open CV2
import time     # библиотека для оценки временных характеристик

#cascadePath = 'tgbot/service/Cascades/haarcascade_frontalface_default.xml' # локальное размещение Модели HAAR
prototxt = 'tgbot/service/Cascades/deploy.prototxt.txt'
model = 'tgbot/service/Cascades/res10_300x300_ssd_iter_140000.caffemodel'
confidence_threshold = 0.5

def dnn_method(in_img, out_img):
    dnn_face_detector = cv2.dnn.readNetFromCaffe(prototxt, model)      # DNN детектор лица
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #image = cv2.imread(in_img) # чтение первоначального изображения

    print("DNN Execution Time (in seconds) :")
    h, w = image.shape[:2]
    Y = h
    str_shape = ' => [' + str(w) + ' x ' + str(h) + ' px]'
    print(str_shape)

    # ---> DNN <---
    start = time.time()
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    dnn_face_detector.setInput(blob) # dnn_net.setInput(blob)
    detections = dnn_face_detector.forward() # detections = dnn_net.forward()
    end = time.time()

    N = str(len(detections))
    boxes = []
    prob = 0.0
    for i in range(0, detections.shape[2]):     # перебор всех лиц
        confidence = detections[0, 0, i, 2]     # вычисление коэффициента доверия обнаруженного лица [0>
        if confidence > confidence_threshold:   # отбивка уровня доверия по confidence_threshold: должн>
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # вычисление геометрии рамки облас>
            (x1, y1, x2, y2) = box.astype("int")
            boxes.append((x1, y1, x2, y2))
            prob = format(confidence * 100, '.2f')
    #    else:
    #        prob = 0.0

    t_dnn = format(end - start, '.2f')
    str_dnn = ' DNN{' + N + '}: ' + str(t_dnn) + ' sec [' + str(prob) + '%]'
    print(str_dnn)

    for item in boxes:
        (x1, y1, x2, y2) = item
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Вывод надписи со всеми временными характеристиками в левый верхний угол # 15 40 60 80 100 125
#    image[Y-125:Y-4, 3:223] = (195,195,127) # заливка области под надписи: координаты [y:y, x:x]
    image[Y-125:Y-4, 3:230] = (195,195,127) # заливка области под надписи: координаты [y:y, x:x]
    cv2.putText(image, str_shape, (5,Y-110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 2)
    cv2.putText(image, str_dnn, (5,Y-90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    is_success, buffer = cv2.imencode(".jpg", image)
    return io.BytesIO(buffer)
#    return in_img
