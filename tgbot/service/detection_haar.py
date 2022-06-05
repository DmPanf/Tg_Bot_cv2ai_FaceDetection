# -*- coding: UTF-8 -*-
# Модуль для тестирования скорости работы и сравнения алгоритмов: HAAR, HOG, CNN

import numpy as np
import io
import cv2      # базовая библиотека Open CV2
import time     # библиотека для оценки временных характеристик

cascadePath = 'tgbot/service/Cascades/haarcascade_frontalface_default.xml' # локальное размещение Модели HAAR

#path1 = sys.argv[1]
#in_img = '../IMG/mfrman.jpg'
#out_img = './img.jpg'

def haar_method(in_img, out_img):
    haar_face_detector = cv2.CascadeClassifier(cascadePath)            # HAAR детектор лица
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #image = cv2.imread(in_img) # чтение первоначального изображения

    print("Execution Time (in seconds) :")
    h, w = image.shape[:2]
    Y = h
    str_shape = ' => [' + str(w) + ' x ' + str(h) + ' px]'
    print(str_shape)

    # ---> HAAR <---
    start = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # преобразование к оттенкам серого
    #faces_haar = haar_face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)) # 1.2
    det_res, rL, lW = haar_face_detector.detectMultiScale3(gray, scaleFactor=1.0485258, minNeighbors=6, minSize=(100,100), outputRejectLevels=1)
    faces_haar = det_res # detection_result
    N = str(len(faces_haar))
    print(rL, lW) # [25] 12.03987 # [rejectLevels] levelWeights
    end = time.time()

    if len(lW) > 0: # levelWeights
        prob = format(100 - lW[0], '.2f')
    else:
        prob = 0.0

    t_haar = format(end - start, '.2f')
    str_haar = 'HAAR{' + N + '}: ' + str(t_haar) + ' sec [' + str(prob) + '%]'
    print(str_haar)

    for (x0,y0,w0,h0) in faces_haar: # для каждого распознанного лица добавляется рамка
        cv2.rectangle(image, (x0,y0), (x0+w0,y0+h0), (255,0,0), 2)

    # Вывод надписи со всеми временными характеристиками в левый верхний угол # 15 40 60 80 100 125
#    image[Y-125:Y-4, 3:223] = (195,195,127) # заливка области под надписи: координаты [y:y, x:x]
    image[Y-125:Y-4, 3:230] = (195,195,127) # заливка области под надписи: координаты [y:y, x:x]
    cv2.putText(image, str_shape, (5,Y-110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 2)
    cv2.putText(image, str_haar, (5,Y-90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    #cv2.imwrite(out_img, image) # сохранить временный файл
    is_success, buffer = cv2.imencode(".jpg", image)
    return io.BytesIO(buffer)

#haar_method(in_img, out_img) # Запускаем функцию с входным и выходным файлом изображения

