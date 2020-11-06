import os
import sys
import time
import yaml

import numpy as np
import soundfile

from tps import cleaners, Handler, load_dict, save_dict
from tps.types import Delimiter

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

import backend_wrappers as bw
from utils.async_utils import BackgroundGenerator
from utils.voice_control import shift_pitch, stretch_wave
from logger import logger

sys.path.pop(0)


def uniqid():
    from time import time
    return hex(int(time() * 1e7))[2:]


_modules_dict = {
    "tacotron2": bw.Tacotron2Wrapper,
    "waveglow": bw.WaveglowWrapper
}


_pauses = {
    Delimiter.eos: 10000,
    Delimiter.semicolon: 5000,
    Delimiter.colon: 3000,
    Delimiter.comma: 2000,
    Delimiter.space: 1000
}


class Synthesizer:
    def __init__(self, name, text_handler, engine, vocoder, sample_rate, device="cuda", pause_type="silence",
                voice_control_cfg=None, user_dict=None):
        self.name = name

        self.text_handler = text_handler
        self.engine = engine
        self.vocoder = vocoder

        self.sample_rate = sample_rate

        self.device = device

        self.pause_type = pause_type
        self.voice_control_cfg = self.load_config(voice_control_cfg)

        self.user_dict = None
        self._dict_source = None
        self.load_user_dict(user_dict)

        logger.info("Synthesizer {} is ready".format(name))


    def synthesize(self, text, **kwargs):
        logger.info(text)

        mask_stress = kwargs.pop("mask_stress", False)
        mask_phonemes = kwargs.pop("mask_phonemes", False)

        sequence = self.text_handler(
            text=text,
            cleaner=cleaners.light_punctuation_cleaners,
            user_dict=self.user_dict,
            keep_delimiters=True,
            mask_stress=mask_stress, mask_phonemes=mask_phonemes
        )

        audio_list = list(self._generate_audio(sequence, **kwargs))
        audio = np.concatenate(audio_list)

        return audio


    def generate(self, text, **kwargs):
        mask_stress = kwargs.pop("mask_stress", False)
        mask_phonemes = kwargs.pop("mask_phonemes", False)

        sequence = self.text_handler.generate(
            text=text,
            cleaner=cleaners.light_punctuation_cleaners,
            user_dict=self.user_dict,
            keep_delimiters=True,
            mask_stress=mask_stress, mask_phonemes=mask_phonemes
        )

        return BackgroundGenerator(self._generate_audio(sequence, **kwargs))


    def _generate_audio(self, sequence, **kwargs):
        logger.debug("kwargs: {}".format(kwargs))

        for unit in sequence:
            if unit in Delimiter:
                duration = _pauses[unit]
                audio = generate_pause(duration, ptype=self.pause_type)
            else:
                logger.debug(unit)
                unit = self.text_handler.check_eos(unit)
                unit = self.text_handler.text2vec(unit)

                spectrogram = self.engine(unit, **kwargs)
                audio = self.vocoder(spectrogram)
                audio = self.vocoder.denoise(audio)

                audio = self.post_process(audio, **kwargs)

            yield audio


    def post_process(self, audio, **kwargs):
        tone_factor = kwargs.pop("tone_factor", None)
        speed_factor = kwargs.pop("speed_factor", None)

        if tone_factor or speed_factor:
            audio = audio.squeeze()
            if tone_factor:
                audio = self.change_pitch(audio, tone_factor)
            if speed_factor:
                audio = self.change_speed(audio, speed_factor)

            audio = self.vocoder.denoise(audio)

        return audio.squeeze()


    def save(self, audio, path, prefix=None):
        os.makedirs(path, exist_ok=True)
        prefix = [prefix] if prefix is not None else []

        waves_format = ".wav"
        name = "_".join(prefix + [self.name, uniqid(), time.strftime("%Y-%m-%d_%H-%M")]) + waves_format

        file_path = os.path.join(path, name)
        soundfile.write(file_path, audio, self.sample_rate)

        logger.info("Audio has been saved as {}".format(os.path.abspath(file_path)))

        return file_path


    def change_speed(self, audio, factor):
        if factor > 2 or factor < 0.5:
            print("ERROR: speed factor is out of range [0.5, 2.0] -- original signal returned")
            return audio

        params = self.voice_control_cfg["phase"]

        return stretch_wave(audio, factor, params)


    def change_pitch(self, audio, factor):
        if factor > 1.5 or factor < 0.75:
            print("ERROR: tone factor is out of range [0.75, 1.5] -- original signal returned")
            return audio

        params = self.voice_control_cfg["psola"]

        return shift_pitch(audio, self.sample_rate, factor, params)


    def load_user_dict(self, user_dict):
        if isinstance(user_dict, dict) or user_dict is None:
            self._dict_source = "./data/{}_user_dict.json".format(self.name)
        else:
            self._dict_source = user_dict
        assert self._dict_source.endswith((".json", ".yaml"))

        self.user_dict = load_dict(user_dict)
        logger.info("User dictionary has been loaded")


    def get_user_dict(self):
        logger.info("Request for the user dictionary has been received")
        return self.user_dict


    def update_user_dict(self, new_dict):
        self.user_dict.update(new_dict)
        logger.info("User dictionary has been updated")

        save_dict(self.user_dict, self._dict_source)
        logger.info("User dictionary has been saved")


    def replace_user_dict(self, new_dict):
        self.user_dict = new_dict
        logger.info("User dictionary has been replaced")

        save_dict(self.user_dict, self._dict_source)
        logger.info("User dictionary has been saved")


    @classmethod
    def from_config(cls, config, name):
        if isinstance(config, str):
            logger.debug("Loading synthesizer from config file {}".format(os.path.abspath(config)))

        config = cls.load_config(config)

        params = config["general"]
        params["name"] = name
        device = params["device"]
        assert device is not None

        modules_config = config.pop(name)
        params["voice_control_cfg"] = modules_config["voice_control_cfg"]
        params["user_dict"] = modules_config["user_dict"]

        params["text_handler"] = _load_text_handler(modules_config["text_handler"])

        chosen = modules_config["modules"]

        for mtype, mname in chosen.items():
            params[mtype] = Synthesizer.module_from_config(modules_config, mtype, mname, device)

        return Synthesizer(**params)


    @staticmethod
    def module_from_config(modules_config, mtype, mname, device):
        logger.info("Loading {} module".format(mname))

        module_config = modules_config[mtype][mname]
        module_config["device"] = device

        for key, value in module_config.pop("options", {}).items():
            if value is not None:
                modules_config[key] = value

        return _modules_dict[mname](**module_config)


    @staticmethod
    def load_config(config_source):
        if isinstance(config_source, dict):
            return config_source
        elif isinstance(config_source, str):
            pass
        else:
            raise TypeError

        with open(config_source, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

        assert config is not None

        return config


def generate_pause(duration, eps=1e-4, ptype='white_noise'):
    if ptype == 'silence':
        pause = np.zeros((duration, ))
    elif ptype == 'white_noise':
        pause = np.random.random((duration, )) * eps
    else:
        raise TypeError

    return pause.astype(np.float32)


def _load_text_handler(config_dict):
    logger.info("Loading text handler")

    out_max_length = config_dict["out_max_length"]

    config_path = config_dict["config_path"]
    assert config_path is not None

    handler_config = Synthesizer.load_config(config_dict["config_path"])
    handler_config["handler"]["out_max_length"] = out_max_length

    return Handler.from_config(handler_config)