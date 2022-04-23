from aiogram import Dispatcher
from aiogram.types import Message, ContentTypes
from aiogram.dispatcher import FSMContext

from aiogram.dispatcher.filters.builtin import Text
from aiogram.utils.markdown import hcode

from tgbot.keyboards import get_method_kb, get_more_kb
from tgbot.states import DetectState

available_methods = ["Сравнение", "HAAR", "HoG", "DNN", "CNN", "MTCNN", "SiAm"]


async def btn_detection(msg: Message):
    await msg.answer("👁‍ Выберите метод обнаружения:", reply_markup=get_method_kb(available_methods))
    await DetectState.waiting_for_method.set()


async def detection_method_chosen(msg: Message, state: FSMContext):
    if msg.text not in available_methods or not msg.text:
        await msg.answer("⚠️ Пожалуйста, выберите метод, используя клавиатуру ниже.")
        return
    await state.update_data(chosen_method=msg.text.lower())

    await DetectState.next()
    await msg.answer("🖼 Теперь прикрепите фотографию")


async def detection_photo_attached(msg: Message, state: FSMContext):
    if not msg.photo:
        await msg.answer("⚠️ Пожалуйста, прикрепите фотографию.")
        return
    user_data = await state.get_data()
    await msg.answer(f"Метод обнаружения: {hcode(user_data['chosen_method'])}\n"
                     f"Фотография: {hcode(msg.photo[-1].file_id)}",
                     reply_markup=get_more_kb("🔳 Обнаружение"))
    await state.finish()


def register_detection(dp: Dispatcher):
    dp.register_message_handler(btn_detection, Text(equals="🔳 Обнаружение"), state="*")
    dp.register_message_handler(detection_method_chosen, state=DetectState.waiting_for_method)
    dp.register_message_handler(detection_photo_attached, content_types=ContentTypes.ANY,
                                state=DetectState.waiting_for_photo)

