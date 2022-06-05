# -*- coding: UTF-8 -*-
# Модуль для тестирования скорости работы и сравнения алгоритмов: HAAR, HOG, CNN
# Необходимо запускать:
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

import numpy as np
import io
import cv2      # базовая библиотека Open CV2
import time     # библиотека для оценки временных характеристик
import mediapipe as mp # MEDIAPIPE – Google AI инструмент для Computer vision []
# детектирования человека, поиск полной маски лица, расположение рук и пальцев, и позу человека

#cascadePath = 'tgbot/service/Cascades/haarcascade_frontalface_default.xml' # локальное размещение Модели HAAR

mp_drawing = mp.solutions.drawing_utils # Initialize the mediapipe drawing class.
mp_face_detection = mp.solutions.face_detection # Initialize the mediapipe face detection class.
Font1 = cv2.FONT_HERSHEY_COMPLEX
Font2 = cv2.FONT_HERSHEY_SIMPLEX
Col1 = (205,205,0)
Col2 = (10,10,10)
ColW = (255,255,255)
ColB = (0,0,0)

def mpdnn_method(in_img, out_img):
    mp_face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.4) # Set up the face detection function by selecting the full-range model.
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #image = cv2.imread(in_img) # чтение первоначального изображения

    print("Execution Time (in seconds) :")
    h, w = image.shape[:2]
    Y = h
    str_shape = ' => [' + str(w) + ' x ' + str(h) + ' px]'
    print(str_shape)

    # ---> MP.DNN <---
    start = time.time()
#    frame = cv2.resize(image, (600, 400))
    faces_mp = mp_face_detector.process(image)
    N = str(len(faces_mp.detections) if faces_mp.detections else 0)
    end = time.time()

    t_mp = format(end - start, '.2f')
    str_mp = 'MPDNN{' + N + '}: ' + str(t_mp) + ' sec'
    print(str_mp)

    if faces_mp.detections:
        for face_no, face in enumerate(faces_mp.detections):
                        # Draw the face bounding box and key points on the copy of the input image.
            mp_drawing.draw_detection(image=image, detection=face, 
                                      keypoint_drawing_spec=mp_drawing.DrawingSpec(color=Col1, thickness=-1, circle_radius=w//115), 
                                      bbox_drawing_spec=mp_drawing.DrawingSpec(color=Col1,thickness=w//180))
            # Retrieve the bounding box of the face.
            face_bbox = face.location_data.relative_bounding_box
            # Retrieve the required bounding box coordinates and scale them according to the size of original input image.
            x1 = int(face_bbox.xmin*w)
            y1 = int(face_bbox.ymin*h)
            prob = str(round(face.score[0], 1))
            # Draw a filled rectangle near the bounding box of the face.
            # We are doing it to change the background of the confidence score to make it easily visible
#            cv2.rectangle(image, pt1=(x1, y1-w//20), pt2=(x1+w//16, y1), color=Col1, thickness=-1) # Закрашенная часть над Рамкой слева
            # Write the confidence score of the face near the bounding box and on the filled rectangle.
#            cv2.putText(image, text=prob, org=(x1, y1-25), fontFace=Font1, fontScale=w//700, color=ColB, thickness=w//200) # Вывод вероятности над Рамкой
    # Вывод надписи со всеми временными характеристиками в левый верхний угол # 15 40 60 80 100 125
#    image[Y-125:Y-4, 3:223] = (195,195,127) # заливка области под надписи: координаты [y:y, x:x]
    image[Y-125:Y-4, 3:235] = (195,195,127) # заливка области под надписи: координаты [y:y, x:x]
    cv2.putText(image, str_shape, (5,Y-110), Font2, 0.5, ColB, 2)
    cv2.putText(image, str_mp, (5,Y-90), Font2, 0.5, ColB, 2)

    is_success, buffer = cv2.imencode(".jpg", image)
    return io.BytesIO(buffer)

#    return in_img
