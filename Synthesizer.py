import os
import re
import time
import yaml

import soundfile

from Utils.Logger import CustomLogger
from Utils.Utils import char_map, sentence_delimiters, PauseTokens
from Utils.VoiceControl import shift_pitch, stretch_wave


def uniqid():
	from time import time
	return hex(int(time() * 1e7))[2:]


class Synthesizer:
	def __init__(self, config, name=None):
		config = self._load_config(config)
		self.cfg_general = config["general"]
		self.cfg_model = config[name]
		self.cfg_voice_control = self._load_config(self.cfg_model["paths"]["voice_control_config"])

		self.name = name

		self.unit_max_length = self.cfg_model["data"]["unit_max_length"]
		self.sentence_as_unit = self.cfg_general["sentence_as_unit"]

		self.logger = CustomLogger(
			logging_level=self.cfg_general["logging_level"],
			log_name=self.name,
			write_to_file=self.cfg_general["log_to_file"],
			style="%(asctime)s - %(name)s - %(levelname)s - %(lineno)s\n%(message)s"
		)


	def synthesize_unit(self, text, **kwargs):
		raise NotImplementedError


	def concatenate(self, units):
		raise NotImplementedError


	def synthesize(self, text, **kwargs):
		sequence = self._prepare_sequence(self._split_to_sentences(text) if self.sentence_as_unit else text)
		sequence = [self._check_eos(item) if item not in PauseTokens else item for item in sequence]

		audio = [self.synthesize_unit(unit, **kwargs) for unit in sequence]

		audio = self.concatenate(audio, **kwargs)

		if self.cfg_general["waves"]["save"]:
			wpath = self.cfg_model["paths"]["waves_folder"]
			file_path = self.save(wpath, audio)

			return audio, file_path

		return audio


	def save(self, wpath, audio):
		if wpath is None:
			self.logger.warning("Path for saving waves is not set, waves won't be written on disk.")
			return

		os.makedirs(wpath, exist_ok=True)

		waves_format = self.cfg_general["waves"]["format"]
		file_path = os.path.join(wpath, "{}_{}_{}.{}".format(self.name, uniqid(), time.strftime("%Y-%m-%d_%H-%M"),
		                                                    waves_format))

		soundfile.write(file_path, audio, self.cfg_general["waves"]["sampling_rate"])

		return file_path


	def change_speed(self, audio, factor):
		if factor > 2 or factor < 0.5:
			print("ERROR: speed factor is out of range [0.5, 2.0] -- original signal returned")
			return audio

		params = self.cfg_voice_control["phase"]

		return stretch_wave(audio, factor, params)


	def change_pitch(self, audio, factor):
		if factor > 1.5 or factor < 0.75:
			print("ERROR: tone factor is out of range [0.75, 1.5] -- original signal returned")
			return audio

		sampling_rate = self.cfg_general["waves"]["sampling_rate"]
		params = self.cfg_voice_control["psola"]

		return shift_pitch(audio, sampling_rate, factor, params)


	def _nlp_preprocessing(self, *args, **kwargs):
		raise NotImplementedError


	def _to_acoustic(self, *args, **kwargs):
		raise NotImplementedError


	def _to_wave(self, *args, **kwargs):
		raise NotImplementedError


	def _split_by_length(self, sentence):
		for char in char_map:
			found = sentence.find(char)
			if found != -1 and found != len(sentence) - 1:
				break

		parts = sentence.split(char)

		part1 = char.join(parts[:len(parts) // 2]) + char
		part2 = char.join(parts[len(parts) // 2:])
		pause = PauseTokens[char_map[char]]

		return [part1, pause, part2]


	def _split_to_sentences(self, text):
		parts = re.split("(?<=[{}]) +".format("".join(sentence_delimiters)), text)

		for i in range(1, len(parts)):
			parts.insert(i * 2 - 1, PauseTokens[char_map["."]])

		return parts


	def _prepare_sequence(self, sentences, subsentence=False):
		sentences = sentences if isinstance(sentences, list) else [sentences]
		sequence = []

		for unit in sentences:
			if unit in PauseTokens:
				if sequence[-1] in PauseTokens:
					sequence.pop()

				sequence.append(unit)
				continue

			if len(unit) <= self.unit_max_length:
				unit = [unit]
			else:
				unit = self._split_by_length(unit)
				unit = self._prepare_sequence(unit, True)

			sequence.extend(unit)

			if not subsentence:
				sequence.append(PauseTokens.sentence_delimiter)

		if sequence[-1] == PauseTokens.sentence_delimiter:
			sequence.pop()

		return sequence


	@staticmethod
	def _check_eos(text):
		raise NotImplementedError


	@staticmethod
	def _load_config(config_source):
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