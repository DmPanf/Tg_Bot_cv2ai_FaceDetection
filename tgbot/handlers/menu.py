# -*- coding: utf-8 -*-
from aiogram import Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.builtin import Text
from aiogram.types import Message
from aiogram.utils.markdown import hcode

from tgbot.keyboards import get_method_kb
from tgbot.states import PhotoState

from tgbot.handlers.new_lvl_menu import available_recognition_methods
from tgbot.handlers.new_lvl_menu import available_detection_methods


async def btn_training(msg: Message):
    await msg.reply("😶‍🌫️ функция в разработке!")


async def btn_control(msg: Message):
    pass


async def btn_recognition(msg: Message):
    await msg.answer("👀 Выберите метод распознования:",
                     reply_markup=get_method_kb(available_recognition_methods))


async def btn_detection(msg: Message):
    await msg.answer("👁‍ Выберите метод обнаружения:",
                     reply_markup=get_method_kb(available_detection_methods))


def register_menu(dp: Dispatcher):
    dp.register_message_handler(btn_recognition, Text(equals="👤 Распознавание"), state=PhotoState)
    dp.register_message_handler(btn_detection, Text(equals="🔳 Обнаружение"), state=PhotoState)
    dp.register_message_handler(btn_detection, Text(equals="🪄 Коррекция"), state=PhotoState)
    dp.register_message_handler(btn_detection, Text(equals="📊 Кластеризация"), state=PhotoState)
    dp.register_message_handler(btn_detection, Text(equals="🔬 Обучение"), state=PhotoState)
