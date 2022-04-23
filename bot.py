#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio

from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.redis import RedisStorage2

from tgbot.config import load_config
from tgbot.handlers.basic import register_basic
from tgbot.handlers.clustering import register_clustering
from tgbot.handlers.correction import register_correction
from tgbot.handlers.detection import register_detection
from tgbot.handlers.recognition import register_recognition
from tgbot.handlers.training import register_training
from tgbot.utils import logger
from tgbot.utils import set_bot_commands


def register_all_handlers(dp):
    register_basic(dp)
    register_detection(dp)
    register_recognition(dp)
    register_correction(dp)
    register_clustering(dp)
    register_training(dp)


async def main():
    logger.info("Starting bot")

    config = load_config(".env")
    storage = RedisStorage2()

    bot = Bot(token=config.tg_bot.token, parse_mode='HTML')
    dp = Dispatcher(bot, storage=storage)

    bot['config'] = config

    register_all_handlers(dp)
    await set_bot_commands(dp)

    # start
    try:
        await dp.start_polling()
    finally:
        await dp.storage.close()
        await dp.storage.wait_closed()
        session = await bot.get_session()
        await session.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.error("Bot stopped!")
