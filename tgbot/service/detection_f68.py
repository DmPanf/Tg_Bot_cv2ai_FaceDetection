# -*- coding: UTF-8 -*-
# Модуль для тестирования скорости работы и сравнения алгоритмов: HAAR, HOG, CNN
#USAGE: python facial_68_Landmark.py

import numpy as np
import io
import cv2      # базовая библиотека Open CV2
import time     # библиотека для оценки временных характеристик
import dlib
from facePoints import facePoints

#cascadePath = 'tgbot/service/Cascades/haarcascade_frontalface_default.xml' # локальное размещение Модели HAAR

Model_PATH = "tgbot/service/Cascades/shape_predictor_68_face_landmarks.dat" # location of the model (path of the model).

def f68_method(in_img, out_img):
# now from the dlib we are extracting the method get_frontal_face_detector()
# and assign that object result to frontalFaceDetector to detect face from the image with
# the help of the 68_face_landmarks.dat model
    frontalFaceDetector = dlib.get_frontal_face_detector()
# Now the dlip shape_predictor class will take model and with the help of that, it will show
    faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #image = cv2.imread(in_img) # чтение первоначального изображения

    print("Execution Time (in seconds) :")
    h, w = image.shape[:2]
    Y = h
    str_shape = ' => [' + str(w) + ' x ' + str(h) + ' px]'
    print(str_shape)

    # ---> Shape_68_faces <---
    start = time.time()
    # Now this line will try to detect all faces in an image either 1 or 2 or more faces
    allFaces = frontalFaceDetector(image, 0)
    end = time.time()

    allFacesLandmark = [] # List to store landmarks of all detected faces
    # Below loop we will use to detect all faces one by one and apply landmarks on them
    for k in range(0, len(allFaces)):
    # dlib rectangle class will detecting face so that landmark can apply inside of that area
        Left = int(allFaces[k].left())
        Top = int(allFaces[k].top())
        Right = int(allFaces[k].right())
        Bottom = int(allFaces[k].bottom())
        faceRectangleDlib = dlib.rectangle(Left, Top, Right, Bottom)
    # Now we are running loop on every detected face and putting landmark on that with the help of faceLandmarkDetector
    detectedLandmarks = faceLandmarkDetector(image, faceRectangleDlib)
    # count number of landmarks we actually detected on image
    if k==0:
        print("Total number of face landmarks detected ",len(detectedLandmarks.parts()))
    # Svaing the landmark one by one to the output folder
    allFacesLandmark.append(detectedLandmarks)
    print(allFacesLandmark)
    # Now finally we drawing landmarks on face
    facePoints(image, detectedLandmarks)
    N = str(len(allFaces) if allFaces else 0)

#            cv2.putText(frame,'FACE',(xmin, (ymin-10)),font, 0.4,(0, 255, 255),1,cv2.LINE_AA)
    t_f68 = format(end - start, '.2f')
    confidence = 0
    str_f68 = ' F68{' + N + '}: ' + str(t_f68) + ' sec [' + str(confidence) + '%]'
    print(str_f68)

    # Вывод надписи со всеми временными характеристиками в левый верхний угол # 15 40 60 80 100 125
#    image[Y-125:Y-4, 3:223] = (195,195,127) # заливка области под надписи: координаты [y:y, x:x]
    image[Y-125:Y-4, 3:230] = (195,195,127) # заливка области под надписи: координаты [y:y, x:x]
    cv2.putText(image, str_shape, (5,Y-110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 2)
    cv2.putText(image, str_f68, (5,Y-90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,50), 2)

    is_success, buffer = cv2.imencode(".jpg", image)
    return io.BytesIO(buffer)

#    return in_img

