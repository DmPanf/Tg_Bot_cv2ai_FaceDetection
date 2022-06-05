# -*- coding: utf-8 -*-
import logging

from aiogram.types import BotCommand

# === LOGGER ===
logger = logging.getLogger(__name__)
formatter = u'[%(asctime)s] %(filename)s:%(lineno)d #%(levelname)-8s %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=formatter,
    datefmt='%H:%M',
)
logger.info("Logger started successfully")


# === MENU ===
async def set_bot_commands(dp):
    await dp.bot.set_my_commands(
        [
            BotCommand("start", "Перезапустить бота"),
            BotCommand("help", "Вывести справку"),
            BotCommand("cancel", "Отменить действие"),
            BotCommand("photo", "Повторить текущее фото"),
        ]
    )

"""
# Делаем отправку инфо в Телеграм
async def send_info(dp):
    await dp.bot.send_message(chat.id=dp.bot.config.tg_bot.ADMINS[0])
"""
