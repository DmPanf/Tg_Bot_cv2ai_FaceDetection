# -*- coding: utf-8 -*-
from aiogram import Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.builtin import Text
from aiogram.types import Message, ContentTypes
from aiogram.utils.markdown import hcode

from tgbot.keyboards import get_more_kb, go_menu_keyboard
from tgbot.states import ClusterState


async def btn_correction(msg: Message):
    await msg.answer("🖼‍ Прикрепите фотографию для кластеризации", reply_markup=go_menu_keyboard())
    await ClusterState.waiting_for_photo.set()


async def correction_photo_attached(msg: Message, state: FSMContext):
    if not msg.photo:
        await msg.answer("⚠️ Пожалуйста, прикрепите фотографию.")
        return
    await msg.answer(f"Фотография: {hcode(msg.photo[-1].file_id)}",
                     reply_markup=get_more_kb("📊 Кластеризация"))
    await state.finish()


def register_clustering(dp: Dispatcher):
    dp.register_message_handler(btn_correction, Text(equals="📊 Кластеризация"), state="*")
    dp.register_message_handler(correction_photo_attached, content_types=ContentTypes.ANY,
                                state=ClusterState.waiting_for_photo)
