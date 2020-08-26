import os
import sys
import time

import numpy as np
import torch
from scipy.io.wavfile import read
from tps import Handler, check_eos

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

from Utils.Utils import PauseTokens, generate_pause
from Synthesizer import Synthesizer, uniqid

sys.path.pop(0)

import_path = os.path.join(root_path, "Backend", "Tacotron2")
sys.path.insert(0, import_path)

from hparams import create_hparams
from model import load_model
from modules.layers import TacotronSTFT

sys.path.pop(0)

import_path = os.path.join(root_path, "Backend", "Waveglow")
sys.path.insert(0, import_path)

from denoiser import Denoiser

sys.path.pop(0)


class Tacotron(Synthesizer):
    def __init__(self, config, name="tacotron2"):
        super().__init__(config, name)

        self.hparams = create_hparams(self.cfg_model["paths"]["hparams"])

        self.device = self.device = torch.device("cpu" if not torch.cuda.is_available() else self.cfg_general["device"])
        self.hparams.device = self.device
        self.dtype = torch.float if self.device.type == "cpu" else torch.half

        self.model = load_model(self.hparams)
        self.model.load_state_dict(torch.load(self.cfg_model["paths"]["model"], map_location=self.device)["state_dict"])
        self.model.eval().to(device=self.device, dtype=self.dtype)

        self.logger.debug("Model has been loaded")

        sys.path.insert(0, import_path)
        self.waveglow = torch.load(self.cfg_model["paths"]["vocoder"], map_location=self.device)["model"]
        sys.path.pop(0)
        self.waveglow.device = self.device

        for m in self.waveglow.modules():
            if "Conv" in str(type(m)):
                setattr(m, "padding_mode", "zeros")

        self.waveglow.eval().to(device=self.device, dtype=self.dtype)

        for k in self.waveglow.convinv:
            k.float()
        self.logger.debug("Vocoder has been loaded")

        self.denoiser = Denoiser(self.waveglow, device=self.device)
        self.logger.debug("Denoiser has been loaded")

        self.stft = TacotronSTFT(
            self.hparams.filter_length, self.hparams.hop_length, self.hparams.win_length,
            self.hparams.n_mel_channels, self.hparams.sampling_rate, self.hparams.mel_fmin,
            self.hparams.mel_fmax)

        self.textHandler = Handler(
            language=self.cfg_model["data"]["language"],
            dictpath=self.cfg_model["paths"]["dict"],
        )

        self.stress = self.cfg_model["data"]["stress"]
        self.phonemes = self.cfg_model["data"]["phonemes"]

        self.ref_audio = self._load_reference(self.cfg_model["paths"]["ref_audio"])


    def synthesize_unit(self, unit, **kwargs):
        if unit in PauseTokens:
            pause = generate_pause(self.cfg_general["duration"][unit.value], type_=self.cfg_general["pause_type"])
            pause = self.stft.mel_spectrogram(torch.from_numpy(pause))
            return pause.to(device=self.device, dtype=self.dtype)

        features = self._nlp_preprocessing(unit)
        spectrogram = self._to_acoustic(features, **kwargs)

        return spectrogram


    def post_process(self, audio, tone_factor, speed_factor):
        if tone_factor or speed_factor:
            audio = audio.squeeze()
            if tone_factor:
                audio = self.change_pitch(audio, tone_factor)
            if speed_factor:
                audio = self.change_speed(audio, speed_factor)

            audio = torch.Tensor(audio).to(device=self.device, dtype=self.dtype)
            audio = self._remove_noises(audio.unsqueeze(0))
            audio = audio.cpu().numpy().astype(np.float32)

        return audio


    def concatenate(self, spectList, **kwargs):
        tone_factor = kwargs.pop("tone_factor", None)
        speed_factor = kwargs.pop("speed_factor", None)

        spectrogram = torch.cat(spectList, dim=2)

        audio = self._to_wave(spectrogram)
        audio = self._remove_noises(audio)

        audio = audio.cpu().numpy().astype(np.float32)
        audio = self.post_process(audio, tone_factor, speed_factor)

        return audio.squeeze()


    def plot_mels(self, text, *mels, save=False, wpath=None):
        import matplotlib.pylab as plt

        fig, axes = plt.subplots(1, len(mels), figsize=(16, 4))
        plt.title(text)

        for i, data in enumerate(mels):
            axes[i].imshow(data, aspect="auto", origin="bottom", interpolation="none")

        if save:
            if wpath is None:
                self.logger.warning("Path for saving plots is not set, plots won't be written on disk.")
            else:
                os.makedirs(wpath, exist_ok=True)
                name = "{}_{}.png".format(self.name, uniqid())
                plt.savefig(os.path.join(wpath, name))


    def reload_reference(self, path):
        self.ref_audio = self._load_reference(path)
        if self.ref_audio is not None:
            self.logger.info("Reference audio has been successfully reloaded")


    def _load_reference(self, path=None):
        if path is None:
            return None

        target_sr = self.cfg_general["waves"]["sampling_rate"]

        sample_rate, audio = read(path)
        audio = np.float32(audio / self.hparams.max_wav_value)

        if sample_rate != target_sr:
            self.logger.warning("{} SR doesn't match target {} SR, can not load audio".format(sample_rate, target_sr))
            return None

        audio = torch.FloatTensor(audio.astype(np.float32))
        audio = audio.unsqueeze(0)
        audio = torch.autograd.Variable(audio)

        mel = self.stft.mel_spectrogram(audio)

        return torch.squeeze(mel, 0).to(device=self.device, dtype=self.dtype)


    def _remove_noises(self, audio):
        if type(audio) == np.ndarray:
            audio = torch.tensor(audio)
        audio = audio.view(1, -1)
        audio = self.denoiser(audio, strength=0.1)[:, 0]
        return audio


    def _nlp_preprocessing(self, text):
        sequence = self.textHandler.text_to_sequence(text, stress=self.stress, phonemes=self.phonemes, dict_prime=False)
        sequence = np.array(sequence)[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence))

        return sequence.to(device=self.device, dtype=torch.long)


    def _to_acoustic(self, features, **kwargs):
        text = kwargs.pop("text", None)

        kwargs.update({
            "max_decoder_steps": int(1500 / 150 * features.size(-1)),
            "reference_mel": self.ref_audio,
        })

        mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(features, **kwargs)

        if self.cfg_general["spectrogram"]["plot"]:
            self.plot_mels(
                text,
                mel_outputs.float().data.cpu().numpy()[0],
                mel_outputs_postnet.float().data.cpu().numpy()[0],
                alignments.float().data.cpu().numpy()[0].T,
                save=self.cfg_general["spectrogram"]["save"],
                wpath=self.cfg_model["paths"]["spectrogram_folder"]
            )

        return mel_outputs_postnet


    def _to_wave(self, spectrogram):
        with torch.no_grad():
            audio = self.waveglow.infer(spectrogram, sigma=0.666)

        return audio


    @staticmethod
    def _check_eos(text):
        return check_eos(text)


def continuous_test_gst():
    tacotron = Tacotron(config="config.yaml")

    while True:
        tacotron.logger.info("Please choose style token {0..9} (-1 for None)")
        tokenIdx = int(input())
        tokenIdx = None if tokenIdx == -1 else tokenIdx

        tacotron.logger.info("Please input your text")
        text = input()

        t1 = time.time()
        tacotron.synthesize(text, token_idx=tokenIdx)
        print("Time spent: {:.1f} sec".format(time.time() - t1))


def continuous_test():
    tacotron = Tacotron(config="Data/config.yaml")

    while True:
        tacotron.logger.info("Please input your text")
        text = input()
        t1 = time.time()
        tacotron.synthesize(text, tone_factor=0.75)
        print("Time spent: {:.1f} sec".format(time.time() - t1))


def main():
    tacotron = Tacotron(config="Data/config.yaml")

    samples = [
        "Съешь же ещё этих мягких французских булок да выпей чаю.",

        "Широкая электрификация южных губерний даст мощный толчок подъёму сельского хозяйства.",

        "В чащах юга жил бы цитрус? Да, но фальшивый экземпляр!",

        "Первый закон робототехники гласит: «Робот не может причинить вред человеку или своим бездействием допустить, "
        "чтобы человеку был причинён вред».",

        "Второй закон робототехники гласит: «Робот должен повиноваться всем приказам, которые дает человек, "
        "кроме тех случаев, когда эти приказы противоречат Первому Закону».",

        "Третий закон робототехники гласит: «Робот должен заботиться о своей безопасности в той мере, "
        "в которой это не противоречит Первому и Второму Законам»."
    ]

    for sample in samples:
        tacotron.synthesize(
            text=sample
        )


if __name__ == "__main__":
    main()