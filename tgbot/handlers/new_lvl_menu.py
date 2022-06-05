# -*- coding: utf-8 -*-
from io import BytesIO
from aiogram import Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.types import Message, InputFile, ChatActions
from aiogram.utils.markdown import hcode, hbold

from tgbot.states import PhotoState
from tgbot.service import detection_haar    # Haar
from tgbot.service import detection_hog     # HoG
from tgbot.service import detection_dnn     # DNN
from tgbot.service import detection_cnn     # CNN
from tgbot.service import detection_mpdnn   # MPDNN
from tgbot.service import detection_f68     # shape_68_faces
from tgbot.service import detection_age     # Age Detection
from tgbot.service import recognition_haar  # HAAR Recognition
from tgbot.service import recognition_cnn   # CNN Recognition
from tgbot.service import smart_avatar      # Smart avatars
from tgbot.service import face_avatar       # Face avatars
from tgbot.service import correction        # Correction
from tgbot.service import clustering        # Clustering


available_detection_methods = ["Сравнение", "HAAR", "DNN", "HoG", "CNN", "MPDNN", "Age_Detect", "Shape_68"]
available_recognition_methods = ["•HAAR", "•CNN"]
available_avatars_methods = ["FACE", "SMART"]
available_cluster_methods = ["K-Means", "Optimal_Bar", "Mean_Bar7", "Color_Pie5", "Main_Bar3"]
available_correction_methods = ["Histogram", "CLAHE", "Red_Eyes", "BW👉Color"]
# Названия кнопок не должны быть одинаковыми

async def new_method_chosen(msg: Message, state: FSMContext):
    if msg.text in available_recognition_methods:
        method = "recognition"
    elif msg.text in available_detection_methods:
        method = "detection"
    elif msg.text in available_avatars_methods:
        method = "avatars"
    elif msg.text in available_cluster_methods:
        method = "cluster"
    elif msg.text in available_correction_methods:
        method = "correction"
    else:
        await msg.answer("⚠️ Пожалуйста, выберите метод, используя клавиатуру ниже.")
        return

    input_image = BytesIO()
    out_image = BytesIO()

    user_data = await state.get_data()
    await msg.bot.download_file_by_id(file_id=user_data['photo_file_id'], destination=input_image)

    out_image = input_image

    photos = []

    try:
        if method == 'detection':
            if msg.text == "HAAR":
                out_image=detection_haar.haar_method(input_image, out_image)
            elif msg.text == "DNN":
                out_image=detection_dnn.dnn_method(input_image, out_image)
            elif msg.text == "HoG":
                out_image=detection_hog.hog_method(input_image, out_image)
            elif msg.text == "CNN":
                out_image=detection_cnn.cnn_method(input_image, out_image)
            elif msg.text == "MPDNN":
                out_image=detection_mpdnn.mpdnn_method(input_image, out_image)
            elif msg.text == "Age_Detect":
                out_image=detection_age.age_method(input_image, out_image)
            elif msg.text == "Shape_68":
                out_image=detection_f68.f68_method(input_image, out_image)

        elif method == 'recognition':
            if msg.text == "•HAAR":
                out_image=recognition_haar.r_haar_method(input_image, out_image)
            elif msg.text == "•CNN":
                out_image=recognition_cnn.r_cnn_method(input_image, out_image)

        elif method == 'avatars':
            if msg.text == "FACE":
                out_image=face_avatar.face_method(input_image, out_image)
            elif msg.text == "SMART":
                out_image=smart_avatar.smart_method(input_image, out_image)

        elif method == 'cluster':
            if msg.text == "K-Means":
                out_image=clustering.kmeans_method(input_image, out_image)
            elif msg.text == "Optimal_Bar":
                out_image=clustering.optimal_method(input_image, out_image)
            elif msg.text == "Mean_Bar7":
                out_image=clustering.colorbar7_method(input_image, out_image)
            elif msg.text == "Color_Pie5":
                out_image=clustering.colorpie_method(input_image, out_image)
            elif msg.text == "Main_Bar3":
                out_image=clustering.colorbar3_method(input_image, out_image)

        elif method == 'correction':
            if msg.text == "Histogram":
                out_image=correction.hist_method(input_image, out_image)
            if msg.text == "CLAHE":
                out_image=correction.clahe_method(input_image, out_image)
            elif msg.text == "Red_Eyes":
                out_image=correction.red_method(input_image, out_image)
            elif msg.text == "BW👉Color":
                out_image=correction.bw2color_method(input_image, out_image)

    except Exception as e:
        print(e)
        out_image = input_image

    # await ChatActions.upload_photo()
    photo = InputFile(path_or_bytesio=out_image)
    Str = f"Task: {hcode(method)}.{hcode(msg.text)}\n"
    await msg.answer_photo(photo=photo, caption=Str)


def register_new_level(dp: Dispatcher):
    dp.register_message_handler(new_method_chosen, state=PhotoState.waiting_for_method)
