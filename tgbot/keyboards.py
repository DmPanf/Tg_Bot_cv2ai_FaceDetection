# -*- coding: utf-8 -*-
from aiogram.types import ReplyKeyboardMarkup


# Main keyboard
def get_start_kb(buttons: list) -> ReplyKeyboardMarkup:
    start_kb = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    return start_kb.add(*buttons)


# Method keyboard
def get_method_kb(buttons: list) -> ReplyKeyboardMarkup:
    detect_kb = ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    return detect_kb.add(*buttons, "↪️ Меню")


def get_more_kb(my_btn: str) -> ReplyKeyboardMarkup:
    detect_more_kb = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    return detect_more_kb.add(*[
        my_btn,
        "↪️ Меню"
    ])


def go_menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(resize_keyboard=True).add("↪️ Меню")
