import glob

import torch
import torchaudio
from torch.utils.data import Dataset

from src.data.converters.converters import BaseConverter


class SoundClassificationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        datasets_to_use: list[str],
        labels_to_use: list[str],
        label2class: dict[str, int],
        content_converter: BaseConverter,
        sample_rate: int = 16000,
    ):
        super().__init__()

        self._sample_rate: int = sample_rate
        
        self._label2class: dict[str, int] = label2class
        self._content_converter: BaseConverter = content_converter

        all_files: list[str] = glob.glob(f'{data_dir}/*/*/*.wav')
        self._files_to_use: list[str] = []

        for file in all_files:
            file_path_parts: list[str] = file.split('/')
            file_dataset: str = file_path_parts[-2]
            file_label: str = file_path_parts[-3]
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

        if sample_rate != self._sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self._sample_rate,
            )

        content: torch.Tensor = self._content_converter.convert(waveform)

        return (content, current_file_class)
