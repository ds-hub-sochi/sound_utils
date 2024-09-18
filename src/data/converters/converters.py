from abc import ABC, abstractmethod

import torch
import torchaudio.transforms as T


class BaseConverter(ABC):
    @abstractmethod
    def convert(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        pass


class WavConverter(BaseConverter):
    def convert(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        return waveform


class SpectrogramConverter(BaseConverter):
    def __init__(
        self,
        n_fft: int,
        normalize: bool = True,
    ):
        super().__init__()

        self._to_spectrogram: T.Spectrogram = T.Spectrogram(
            n_fft=n_fft,
            normalized=normalize,
        )

    def convert(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        return self._to_spectrogram(waveform)


class MelSpectrogramConverter(BaseConverter):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
    ):
        super().__init__()

        self._to_mel_spectrogram: T.MelSpectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            center=True,
            pad_mode='reflect',
            power=2.0,
            norm='slaney',
            mel_scale='htk',
        )

    def convert(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        return self._to_mel_spectrogram(waveform)


class MelSpectrogramDBConverter(MelSpectrogramConverter):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
    ):
        super().__init__(
            sample_rate,
            n_fft,
            n_mels,
        )

        self._to_db = T.AmplitudeToDB()

    def convert(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        return self._to_db(self._to_mel_spectrogram(waveform))


class MFCCConverter(BaseConverter):
    def __init__(
        self,
        sample_rate: int,
        n_mfcc: int,
        n_fft: int,
        n_mels: int,
    ):
        super().__init__()

        self._to_mfcc: T.MFCC = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'n_mels': n_mels,
                'mel_scale': 'htk',
            },
        )

    def convert(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        return self._to_mfcc(waveform)


class LFCCConverter(BaseConverter):
    def __init__(
        self,
        sample_rate: int,
        n_lfcc: int,
        n_fft: int,
    ):
        super().__init__()

        self._to_lfcc: T.LFCC = T.LFCC(
            sample_rate=sample_rate,
            n_lfcc=n_lfcc,
            speckwargs={
                'n_fft': n_fft,
            },
        )

    def convert(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        return self._to_lfcc(waveform)
