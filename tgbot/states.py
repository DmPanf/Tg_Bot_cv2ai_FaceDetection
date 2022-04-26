# -*- coding: utf-8 -*-
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.dispatcher import FSMContext


class PhotoState(StatesGroup):
    waiting_for_method = State()


class ControlState(StatesGroup):
    pass
