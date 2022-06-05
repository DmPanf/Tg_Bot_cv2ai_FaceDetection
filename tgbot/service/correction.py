# -*- coding: UTF-8 -*-

import numpy as np
import io
import cv2

cascadePath = 'tgbot/service/Cascades/haarcascade_eye.xml' # Red Eyes
eyesCascade = cv2.CascadeClassifier(cascadePath)
# BW 2 Color
prototxt = "tgbot/service/Cascades/colorization_deploy_v2.prototxt"
model = "tgbot/service/Cascades/colorization_release_v2.caffemodel"
points = "tgbot/service/Cascades/pts_in_hull.npy"

def clahe_method(in_img, out_img):
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#-----Converting image to LAB Color model-----------------------------------
#    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#-----Splitting the LAB image to different channels-------------------------
#    l, a, b = cv2.split(lab)
#-----Applying CLAHE to L-channel-------------------------------------------
#    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#    cl = clahe.apply(l)
#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
#    limg = cv2.merge((cl,a,b))
#-----Converting image from LAB Color model to RGB model--------------------
#    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Resizing the image for compatibility
    #image = cv2.resize(image, (500, 600))
    # The initial processing of the image
    #image = cv2.medianBlur(image, 3)
    # image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Ver.1 The declaration of CLAHE # clipLimit -> Threshold for contrast limiting
    # clahe = cv2.createCLAHE(clipLimit = 5)
    # clahe_img = clahe.apply(image_bw) + 30
    # Ver.2 create a CLAHE object (Arguments are optional).
    # clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
    # clahe_img = clahe.apply(image)
    # Ordinary thresholding the same image
    #_, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)
    # Showing the image

    # Ver.3 CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2,a,b))  # merge channels
    clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

    is_success, buffer = cv2.imencode(".jpg", clahe_img)
    return io.BytesIO(buffer)


def hist_method(in_img, out_img):
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_to_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

    is_success, buffer = cv2.imencode(".jpg", hist_equalization_result)
    return io.BytesIO(buffer)


# --------------> Глобальное выравнивание гистограммы <--------------------------------------
#def equalHist(img):
#    # Высота и ширина матрицы серого изображения
#    h, w = img.shape
#    # Шаг 1: Рассчитать гистограмму в градациях серого
#    grayHist = calcGrayHist(img)
#    # Шаг 2: Рассчитать накопленную серую гистограмму
#    zeroCumuMoment = np.zeros([256], np.uint32)
#    for p in range(256):
#        if p == 0:
#            zeroCumuMoment[p] = grayHist[0]
#        else:
#            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
#    # Шаг 3: Получить соотношение отображения между уровнем серого на входе и уровнем серого на основе накопленной гистограммы серого
#    outPut_q = np.zeros([256], np.uint8)
#    cofficient = 256.0 / (h * w)
#    for p in range(256):
#        q = cofficient * float(zeroCumuMoment[p]) - 1
#        if q >= 0:
#            outPut_q[p] = math.floor(q)
#        else:
#            outPut_q[p] = 0
#    # Четвертый шаг: получить выровненное изображение гистограммы
#    equalHistImage = np.zeros(img.shape, np.uint8)
#    for i in range(h):
#        for j in range(w):
#            equalHistImage[i][j] = outPut_q[img[i][j]]
#    return equalHistImage

# ----------> Red Eyes <----------

def fillHoles(mask):
    maskFloodfill = mask.copy()
    h, w = maskFloodfill.shape[:2]
    maskTemp = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
    mask2 = cv2.bitwise_not(maskFloodfill)
    return mask2 | mask



def red_method(in_img, out_img):
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    imgOut = image.copy() # Output image
    eyes = eyesCascade.detectMultiScale(image,scaleFactor=1.3, minNeighbors=4, minSize=(100, 100)) # Detect eyes
    #print(eyes)
    for (x, y, w, h) in eyes:
        eye = image[y:y+h, x:x+w]  # Extract eye from the image.
        # Split eye image into 3 channels
        b = eye[:, :, 0]
        g = eye[:, :, 1]
        r = eye[:, :, 2]
        bg = cv2.add(b, g)  # Add the green and blue channels.
        mask = (r > 150) &  (r > bg)  # Simple red eye detector
        mask = mask.astype(np.uint8)*255  # Convert the mask to uint8 format.
        # Clean up mask by filling holes and dilating
        mask = fillHoles(mask)
        mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)
        # Calculate the mean channel by averaging the green and blue channels. Recall, bg = cv2.add(b, g)
        mean = bg / 2
        mask = mask.astype(np.bool)[:, :, np.newaxis]
        mean = mean[:, :, np.newaxis]
        eyeOut = eye.copy() # Copy the eye from the original image.
        eyeOut = eyeOut.astype('float64') # Иначе: Cannot cast array data from dtype('float64') to dtype('uint8') according to the rule 'same_kind'
        np.copyto(eyeOut, mean, where=mask) # Copy the mean image to the output image.
        #print('OK' + x)
        imgOut[y:y+h, x:x+w, :] = eyeOut # Copy the fixed eye to the output image

    #result = np.hstack((in_img, imgOut))
    is_success, buffer = cv2.imencode(".jpg", imgOut)
    return io.BytesIO(buffer)

# --------------------- BW -> Color <----------------------------------
# нейросеть работает с изображениями 224х224
# нейросеть была обучена на 1.3млн изображений

def load_model():
    # Load serialized black and white colorizer model and cluster
    # The L channel encodes lightness intensity only
    # The a channel encodes green-red.
    # And the b channel encodes blue-yellow
    print("Loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    # Add the cluster centers as 1x1 convolutions to the model:
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net

def colorize_image(net, image):
    # Load the input image, scale it and convert it to Lab:
#    image = cv2.imread(image_in)
    height, width, channels = image.shape
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Extracting "L"
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    # Resize to network size
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    # Predicting "a" and "b"
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    # Creating a colorized Lab photo (L + a + b)
    L = cv2.split(lab)[0]
    ab = cv2.resize(ab, (width, height))
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    # Convert to RGB
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    image_out = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)
    return image_out

def bw2color_method(in_img, out_img):
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    net = load_model()
    new_img = colorize_image(net, image)
#    result = np.hstack((in_img, new_img))
#    is_success, buffer = cv2.imencode(".jpg", result)
    is_success, buffer = cv2.imencode(".jpg", new_img)
    return io.BytesIO(buffer)

