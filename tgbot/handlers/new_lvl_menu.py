# -*- coding: utf-8 -*-
from aiogram import Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.types import Message
from aiogram.utils.markdown import hcode, hbold

from tgbot.states import PhotoState

available_detection_methods = ["Сравнение", "HAAR", "HoG", "DNN", "CNN", "MTCNN", "SiAm"]
available_recognition_methods = ["•HAAR", "•CNN"]


async def new_method_chosen(msg: Message, state: FSMContext):
    if msg.text in available_recognition_methods:
        method = "recognition"
    elif msg.text in available_detection_methods:
        method = "detection"
    else:
        await msg.answer("⚠️ Пожалуйста, выберите метод, используя клавиатуру ниже.")
        return

    user_data = await state.get_data()
    await msg.answer(f"Task: {hbold(method)}\n"
                     f"Method: {hbold(msg.text)}\n"
                     f"Photo_id: {hcode(user_data['photo_file_id'])}")


def register_new_level(dp: Dispatcher):
    dp.register_message_handler(new_method_chosen, state=PhotoState)
