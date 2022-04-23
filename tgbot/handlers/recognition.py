# -*- coding: utf-8 -*-
from aiogram import Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.builtin import Text
from aiogram.types import Message, ContentTypes
from aiogram.utils.markdown import hcode

from tgbot.keyboards import get_method_kb, get_more_kb
from tgbot.states import RecognizeState

available_methods = ["HAAR", "CNN"]


async def btn_recognition(msg: Message):
    await msg.answer("👀 Выберите метод распознования:", reply_markup=get_method_kb(available_methods))
    await RecognizeState.waiting_for_method.set()


async def recognition_method_chosen(msg: Message, state: FSMContext):
    if msg.text not in available_methods or not msg.text:
        await msg.answer("⚠️ Пожалуйста, выберите метод, используя клавиатуру ниже.")
        return
    await state.update_data(chosen_method=msg.text.lower())

    await RecognizeState.next()
    await msg.answer("🖼 Теперь прикрепите фотографию")


async def recognition_photo_attached(msg: Message, state: FSMContext):
    if not msg.photo:
        await msg.answer("⚠️ Пожалуйста, прикрепите фотографию.")
        return
    user_data = await state.get_data()
    await msg.answer(f"Метод распознавания: {hcode(user_data['chosen_method'])}\n"
                     f"Фотография: {hcode(msg.photo[-1].file_id)}",
                     reply_markup=get_more_kb("👤 Распознавание"))
    await state.finish()


def register_recognition(dp: Dispatcher):
    dp.register_message_handler(btn_recognition, Text(equals="👤 Распознавание"), state="*")
    dp.register_message_handler(recognition_method_chosen, state=RecognizeState.waiting_for_method)
    dp.register_message_handler(recognition_photo_attached, content_types=ContentTypes.ANY,
                                state=RecognizeState.waiting_for_photo)
