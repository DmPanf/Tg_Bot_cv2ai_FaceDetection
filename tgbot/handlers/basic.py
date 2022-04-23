# -*- coding: utf-8 -*-
from aiogram import Dispatcher
from aiogram.types import Message
from aiogram.dispatcher import FSMContext

from aiogram.dispatcher.filters.builtin import CommandStart, CommandHelp, Text

from tgbot.keyboards import get_start_kb


available_buttons = ["🔳 Обнаружение", "👤 Распознавание", "🪄 Коррекция", "📊 Кластеризация", "⚙️ Управление", "🔬 Обучение"]


# Start command
async def cmd_start(msg: Message, state: FSMContext):
    await state.finish()
    await msg.answer(f"<b>👤 Привет, {msg.from_user.full_name}!</b>\n\n",
                     reply_markup=get_start_kb(available_buttons)
                     )


# Cancel command
async def cmd_cancel(msg: Message, state: FSMContext):
    await state.finish()
    await msg.answer("Действие отменено", reply_markup=get_start_kb())


# Main menu
async def cmd_menu(msg: Message, state: FSMContext):
    await state.finish()
    await msg.answer("Открыто главное меню", reply_markup=get_start_kb(available_buttons))


# Help command
async def cmd_help(msg: Message, state: FSMContext):
    await state.finish()
    config = msg.bot.get('config')
    await msg.answer(f"{config.settings.help_msg}")


def register_basic(dp: Dispatcher):
    dp.register_message_handler(cmd_start, CommandStart(), state="*")
    dp.register_message_handler(cmd_help, CommandHelp(), state="*")

    dp.register_message_handler(cmd_cancel, commands="cancel", state="*")
    dp.register_message_handler(cmd_cancel, Text(equals="отмена", ignore_case=True), state="*")

    dp.register_message_handler(cmd_menu, commands="menu", state="*")
    dp.register_message_handler(cmd_menu, Text(equals="↪️ Меню"), state="*")
