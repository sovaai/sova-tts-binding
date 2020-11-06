import sys
from loguru import logger


# sys.stdout.reconfigure(encoding="utf-8")
class Format:
    time = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>"
    level = "<level>{level: <8}</level>"
    module = "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
    message = "<level>{message}</level>"

LEVEL = "INFO"
SAVE_TO_FILE = False
_filename = "data/logs/user.log"

logger.remove()
logger.add(sys.stdout, level=LEVEL)

if SAVE_TO_FILE:
    logger.add(_filename, encoding="utf8")


# from collections import defaultdict
# from random import choice
#
# colors = ["blue", "green", "magenta", "red", "yellow"]
# color_per_module = defaultdict(lambda: choice(colors))
#
# logger.bind(synthesizer_name=name)
# _color_tag = choice(colors)
# _name_fmt = "<{}>".format(_color_tag) + "{extra[synthesizer_name]}" + "</{}>".format(_color_tag)
# _formatter = " | ".join((Format.time, Format.level, Format.module, _name_fmt, Format.message))
# logger.add(sys.stdout, format=_formatter)