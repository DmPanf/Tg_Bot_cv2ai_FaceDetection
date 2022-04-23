# -*- coding: utf-8 -*-
from aiogram.dispatcher.filters.state import StatesGroup, State


class TestState(StatesGroup):
    location = State()


class DetectState(StatesGroup):
    waiting_for_method = State()
    waiting_for_photo = State()


class RecognizeState(StatesGroup):
    waiting_for_method = State()
    waiting_for_photo = State()


class CorrectState(StatesGroup):
    waiting_for_photo = State()


class ClusterState(StatesGroup):
    waiting_for_photo = State()


class ControlState(StatesGroup):
    pass
