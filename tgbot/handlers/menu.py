# -*- coding: utf-8 -*-
import io

from aiogram import Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.builtin import Text
from aiogram.types import Message, CallbackQuery, InputFile, ChatActions
from datetime import datetime

from tgbot.keyboards import get_method_kb, get_control_kb
from tgbot.states import PhotoState

from tgbot.handlers.new_lvl_menu import available_recognition_methods
from tgbot.handlers.new_lvl_menu import available_detection_methods


async def btn_training(msg: Message):
    await msg.reply("😶‍🌫️ функция в разработке!")


async def btn_control(msg: Message):
    await msg.answer(f"<b>⚙️ Панель управления</b> [{datetime.utcnow().strftime('%d.%m - %H:%M')}]",
                     reply_markup=get_control_kb())


async def btn_recognition(msg: Message):
    await msg.answer("👀 Выберите метод распознования:",
                     reply_markup=get_method_kb(available_recognition_methods))


async def btn_detection(msg: Message):
    await msg.answer("👁‍ Выберите метод обнаружения:",
                     reply_markup=get_method_kb(available_detection_methods))


async def btn_correction(msg: Message, state: FSMContext):
    user_data = await state.get_data()
    await msg.answer(f"{user_data['photo_file_id']}")


async def btn_clustering(msg: Message, state: FSMContext):
    user_data = await state.get_data()
    await msg.answer(f"{user_data['photo_file_id']}")


async def cb_response(call: CallbackQuery):
    print("Ok")
    await call.answer(
        text=f"❎ Функция в разработке",
        cache_time=3,
        show_alert=False
    )


def register_menu(dp: Dispatcher):
    dp.register_message_handler(btn_recognition, Text(equals="👤 Распознавание"), state=PhotoState)
    dp.register_message_handler(btn_detection, Text(equals="🔳 Обнаружение"), state=PhotoState)
    dp.register_message_handler(btn_correction, Text(equals="🪄 Коррекция"), state=PhotoState.waiting_for_method)
    dp.register_message_handler(btn_clustering, Text(equals="📊 Кластеризация"), state=PhotoState.waiting_for_method)
    dp.register_message_handler(btn_training, Text(equals="🔬 Обучение"), state=PhotoState)
    dp.register_message_handler(btn_control, Text(equals="⚙️ Управление"), state=PhotoState)

    dp.register_callback_query_handler(cb_response, state="*")