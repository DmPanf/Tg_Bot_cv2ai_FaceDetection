# -*- coding: utf-8 -*-
from dataclasses import dataclass
from configparser import ConfigParser
from environs import Env


@dataclass
class TgBot:
    token: str
    admin_ids: list
    use_redis: bool


@dataclass
class Settings:
    help_msg: str = None
    other_param: str = None


@dataclass
class Config:
    tg_bot: TgBot
    settings: Settings


def load_config(path: str = None):
    env = Env()
    env.read_env(path)

    ini_reader = ConfigParser()
    ini_reader.read("settings.ini")

    return Config(
        tg_bot=TgBot(
            token=env.str("TG_TOKEN"),
            admin_ids=list(map(int, env.list("ADMINS"))),
            use_redis=env.bool("USE_REDIS"),
        ),
        settings=Settings(
            help_msg=ini_reader.get('HELP', 'msg'),
            other_param=ini_reader.get('OTHER', 'param')
        )
    )
