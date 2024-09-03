import glob

import torchaudio
from torch.utils.data import Dataset


class WavSoundClassificationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        label2class: dict[str, int],
        labels_to_use: list[str],
        datasets_to_use: list[str],
        sample_rate_to_use: int = 16000,
    ):
        super().__init__()

        self._sample_rate_to_use: int = sample_rate_to_use

        self._label2class: dict[str, int] = label2class

        all_files: list[str] = glob.glob(f'{data_dir}/*/*.wav')
        self._files_to_use: list[str] = []

        for file in all_files:
            file_dataset: str = file.split('/')[-2]
            file_label: str = file.split('/')[-3]
            if file_dataset in datasets_to_use and file_label in labels_to_use:
                self._files_to_use.append(file)

    def __len__(
        self,
    ) -> int:
        return len(self._files_to_use)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:
        path_to_current_file: str = self._files_to_use[index]

        current_file_class: int = self._label2class[path_to_current_file.split('/')[-3]]

        waveform, sample_rate = torchaudio.load(
            path_to_current_file,
            backend='ffmpeg',
            normalize=True,
        )

        if sample_rate != sample_rate_to_use:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=sample_rate_to_use,
            )

        return (waveform, current_file_class)


class SoundSpectrogramDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        label2class: dict[str, int],
        labels_to_use: list[str],
        datasets_to_use: list[str],
        n_mels: int = 64,
        sample_rate_to_use: int = 16000,
    ):
        super().__init__()

        self._sample_rate_to_use: int = sample_rate_to_use

        self._label2class: dict[str, int] = label2class

        all_files: list[str] = glob.glob(f'{data_dir}/*/*.wav')
        self._files_to_use: list[str] = []

        for file in all_files:
            file_dataset: str = file.split('/')[-2]
            file_label: str = file.split('/')[-3]
            if file_dataset in datasets_to_use and file_label in labels_to_use:
                self._files_to_use.append(file)

        self._n_mels: int = n_mels

    def __len__(
        self,
    ) -> int:
        return len(self._files_to_use)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:
        path_to_current_file: str = self._files_to_use[index]

        current_file_class = self._label2class[path_to_current_file.split('/')[-3]]

        waveform, sample_rate = torchaudio.load(
            path_to_current_file,
            backend='ffmpeg',
            normalize=True,
        )

        if sample_rate != sample_rate_to_use:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=sample_rate_to_use,
            )

        melspectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate_to_use,
            n_mels=self._n_mels,
        )
        melspectrogram = melspectrogram_transform(waveform)
        melspectogram_db = torchaudio.transforms.AmplitudeToDB()(melspectrogram)

        return (melspectogram_db, current_file_class)
