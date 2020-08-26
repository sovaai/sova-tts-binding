from collections import OrderedDict
from enum import Enum

import numpy as np

char_map = OrderedDict({
    ".": "sentence_delimiter",
    "?": "sentence_delimiter",
    "!": "sentence_delimiter",
    ":": "colon",
    ";": "semicolon",
    ",": "comma",
    " ": "space"
})

sentence_delimiters = [".", "?", "!"]


class PauseTokens(str, Enum):
    sentence_delimiter = "sentence_delimiter"
    colon = "colon"
    semicolon = "semicolon"
    comma = "comma"
    space = "space"


def generate_pause(duration, eps=1e-4, type_='white_noise'):
    if type_ == 'silence':
        pause = np.zeros((1, duration))
    elif type_ == 'white_noise':
        pause = np.random.random((1, duration)) * eps
    else:
        raise TypeError

    return pause.astype(np.float32)
