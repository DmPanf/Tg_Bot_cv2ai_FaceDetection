# -*- coding: UTF-8 -*-
# Модуль для тестирования скорости работы и сравнения алгоритмов: HAAR, HOG, CNN

import numpy as np
import io
import cv2      # базовая библиотека Open CV2
import time     # библиотека для оценки временных характеристик


# define the list of age buckets our age detector will predict
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12 or 16)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
genderList=['Male', 'Female']
Conf = 0.5

#cascadePath = 'tgbot/service/Cascades/haarcascade_frontalface_default.xml' # локальное размещение Модели HAAR

# load our serialized face detector model from disk
faceProto = 'tgbot/service/Cascades/deploy.prototxt.txt' # prototxtPath
faceModel = 'tgbot/service/Cascades/res10_300x300_ssd_iter_140000.caffemodel' # weights

# load our serialized age detector model from disk
ageProto = 'tgbot/service/Cascades/age_deploy.prototxt' # prototxtPath
ageModel = 'tgbot/service/Cascades/age_net.caffemodel' # weights

# Gender Detector Model
genderProto = 'tgbot/service/Cascades/gender_deploy.prototxt' # prototxtPath
genderModel = 'tgbot/service/Cascades/gender_net.caffemodel' # weights

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746) # Olds: (104, 117, 123)
SIZE = (227,227) # (224, 224)
Col0 = (127, 2555, 0) # Основной цвет рамки ROI и надписей
Font = cv2.FONT_HERSHEY_SIMPLEX
pad = 20

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), Col0, int(round(frameHeight/150)), 3)
    return frameOpencvDnn,faceBoxes

def age_method(in_img, out_img):
    print("[INFO] loading Face detector model...")
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    print("[INFO] loading Age detector model...")
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    print("[INFO] loading Gender detector model...")
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #image = cv2.imread(in_img) # чтение первоначального изображения

    print("Execution Time (in seconds) :")
    (h, w) = image.shape[:2] # load the input image and construct an input blob for the image
    Y = h
    str_shape = ' => [' + str(w) + ' x ' + str(h) + ' px]'
    print(str_shape)

    # ---> Age DNN Detection <---
    start = time.time()
    # pass the blob through the network and obtain the face detections
#    blob = cv2.dnn.blobFromImage(image, 1.0, SIZE, (104, 117, 123)) # scalefactor=1.0 # Olds: (104, 117, 123)
#    print("[INFO] computing face detections...")
#    faceNet.setInput(blob)
#    detections = faceNet.forward()

    resultImg, faceBoxes = highlightFace(faceNet, image)
    if not faceBoxes:
        print("No face detected!")

    for faceBox in faceBoxes:
        face = image[max(0, faceBox[1]-pad):min(faceBox[3]+pad, image.shape[0]-1), 
                   max(0, faceBox[0]-pad):min(faceBox[2]+pad, image.shape[1]-1)]
        faceBlob = cv2.dnn.blobFromImage(face, 1.0, SIZE, MODEL_MEAN_VALUES, swapRB=False)

#    for i in range(0, detections.shape[2]): # loop over the detections
#        confidence = detections[0, 0, i, 2] # extract the confidence (i.e., probability) associated with the prediction
#        if confidence > Conf:  # filter out weak detections by ensuring the confidence is greater than the minimum confidence
#            prob = "Confidence: {:.2f}%".format(confidence * 100) # "{}: {:.2f}%".format(age, ageConfidence * 100)
#            print(prob) # "{}: {:.2f}%".format(age, ageConfidence * 100)
#            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # compute the (x, y)-coordinates of the bounding box for the object
#            (startX, startY, endX, endY) = box.astype("int")
#            face = image[startY:endY, startX:endX] # extract the ROI of the face and then construct a blob from *only* the face ROI
#            faceBlob = cv2.dnn.blobFromImage(face, 1.0, SIZE, MODEL_MEAN_VALUES, swapRB=False) # MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)

        ageNet.setInput(faceBlob) # make predictions on the age and find the age bucket with the largest corresponding probability
        agePreds = ageNet.forward() # age=ageList[agePreds[0].argmax()]
        j = agePreds[0].argmax()
        age = AGE_BUCKETS[j]
        ageConfidence = agePreds[0][j]
        print(f'Age: {age[1:-1]} years')

        genderNet.setInput(faceBlob)
        genderPreds = genderNet.forward()
        z = genderPreds[0].argmax()
        gender = genderList[z] # gender = genderList[genderPreds[0].argmax()]
        genderConfidence = genderPreds[0][z]
        print(f'Gender: {gender}')

        ageStr = "{}: {:.2f}%".format(age, ageConfidence * 100)
        genderStr = "{}: {:.2f}%".format(gender, genderConfidence * 100)
        text = ageStr + ' \ ' + genderStr
        print("[INFO] {}".format(text)) # display the predicted age to the terminal

        cv2.putText(resultImg, genderStr, (faceBox[0], faceBox[1]-35), Font, 0.95, Col0, 2, cv2.LINE_AA)
        cv2.putText(resultImg, ageStr, (faceBox[0], faceBox[1]-10), Font, 0.95, Col0, 2, cv2.LINE_AA)

#        y = startY - 10 if startY - 10 > 10 else startY + 10 # draw the bounding box of the face along with the associated predicted age
#        cv2.rectangle(image, (startX, startY), (endX, endY), Col0, 3)
#        cv2.putText(image, ageStr, (startX, y), Font, 0.95, Col0, 3)
#        cv2.putText(image, genderStr, (startX, endY + 25), Font, 0.95, Col0, 3)

    end = time.time()
    t_age = format(end - start, '.2f')
    str_age = 'Time: ' + str(t_age) + ' sec'
    print(str_age)

    is_success, buffer = cv2.imencode(".jpg", resultImg) # image
    return io.BytesIO(buffer)

#    return in_img
