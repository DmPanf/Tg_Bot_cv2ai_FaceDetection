# -*- coding: utf-8 -*-
from aiogram.types import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton


# Main keyboard
def get_start_kb(buttons: list) -> ReplyKeyboardMarkup:
    start_kb = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    return start_kb.add(*buttons)


# Method keyboard
def get_method_kb(buttons: list) -> ReplyKeyboardMarkup:
    detect_kb = ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    return detect_kb.add(*buttons, "↪️ Меню")


def get_control_kb() -> InlineKeyboardMarkup:
    control_kb = InlineKeyboardMarkup(row_width=2)
    btns = [
        InlineKeyboardButton("🛠 Администрирование", callback_data="control:admin"),
        InlineKeyboardButton("📈 Статистика", callback_data="control:stat"),
        InlineKeyboardButton("🎛 Датчики", callback_data="control:sensors"),
        InlineKeyboardButton("💾 Фото-архив", callback_data="control:photo_archive"),
        InlineKeyboardButton("🚪 Дверь", callback_data="control:door"),
        InlineKeyboardButton("🪟 Жалюзи", callback_data="control:window"),
    ]
    control_kb.add(*btns)
    control_kb.row(InlineKeyboardButton("📂 Код бота", url="https://github.com/dnp34/cv2ai"))
    return control_kb
