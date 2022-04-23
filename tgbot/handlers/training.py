# -*- coding: utf-8 -*-
from aiogram import Dispatcher
from aiogram.dispatcher.filters.builtin import Text
from aiogram.types import Message


async def btn_training(msg: Message):
    await msg.answer("😶‍🌫️ функция <b>«Обучение»</b> на данный момент в разработке!")


def register_training(dp: Dispatcher):
    dp.register_message_handler(btn_training, Text(equals="🔬 Обучение"), state="*")
