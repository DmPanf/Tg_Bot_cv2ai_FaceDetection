#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio

from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.redis import RedisStorage2

from tgbot.config import load_config
from tgbot.handlers.basic import register_basic
from tgbot.handlers.menu import register_menu
from tgbot.handlers.new_lvl_menu import register_new_level

from tgbot.utils import logger
from tgbot.utils import set_bot_commands


def register_all_handlers(dp):
    register_basic(dp)
    register_menu(dp)
    register_new_level(dp)


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
