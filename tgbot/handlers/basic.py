# -*- coding: utf-8 -*-
from aiogram import Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.builtin import CommandStart, CommandHelp, Text
from aiogram.types import Message, ContentTypes, ReplyKeyboardRemove

from tgbot.keyboards import get_start_kb
from tgbot.states import PhotoState

available_buttons = ["🔳 Обнаружение", "👤 Распознавание", "🪄 Коррекция", "📊 Кластеризация", "⚙️ Управление",
                     "🔬 Обучение"]


# Start command
async def cmd_start(msg: Message, state: FSMContext):
    await state.finish()
    await msg.answer(f"<b>👤 Здравствуйте, {msg.from_user.full_name}!</b>\n\nДля начала отправьте фото",
                     reply_markup=ReplyKeyboardRemove())


# Cancel command
async def cmd_cancel(msg: Message):
    await msg.answer("Что сделать с фото?", reply_markup=get_start_kb(available_buttons))


# Help command
async def cmd_help(msg: Message):
    config = msg.bot.get('config')
    await msg.answer(f"{config.settings.help_msg}")


#
async def photo_handler(msg: Message, state: FSMContext):
    await state.set_state(PhotoState.waiting_for_method)
    await state.update_data(photo_file_id=msg.photo[-1].file_id)
    await msg.reply("Test", reply_markup=get_start_kb(available_buttons))


def register_basic(dp: Dispatcher):
    dp.register_message_handler(cmd_start, CommandStart(), state="*")
    dp.register_message_handler(cmd_help, CommandHelp(), state="*")

    dp.register_message_handler(cmd_cancel, commands="cancel", state=PhotoState)
    dp.register_message_handler(cmd_cancel, Text(equals="отмена", ignore_case=True), state=PhotoState)
    dp.register_message_handler(cmd_cancel, Text(equals="↪️ Меню"), state=PhotoState)

    dp.register_message_handler(photo_handler, content_types=ContentTypes.PHOTO, state="*")
