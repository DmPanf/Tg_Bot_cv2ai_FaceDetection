# -*- coding: UTF-8 -*-

import numpy as np
import io
import cv2

cascadePath = 'tgbot/service/Cascades/..'


def r_cnn_method(in_img, out_img):
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    is_success, buffer = cv2.imencode(".jpg", image)
    return io.BytesIO(buffer)

