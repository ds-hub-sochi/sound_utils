import pathlib
import typing as tp
from math import ceil

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm


def sort_by_filename(
    label2content: dict[str, dict[str, list[torch.Tensor]]],
) -> None:
    for label in label2content:
        label2content[label] = dict(sorted(label2content[label].items()))


def print_number_of_sample_for_classes(
    label2content: dict[str, dict[str, list[torch.Tensor]]],
) -> None:
    for label in label2content:
        counter: int = 0
        for filename in label2content[label]:
            counter += len(label2content[label][filename])
        print(f'{label}: {counter}')


def fill_train_val_test_dirs(
    label2content: dict[str, dict[str, list[torch.Tensor]]],
    sample_rate: int,
    train_size: float,
    validation_size: float,
    additional_label: str,
    train_dump_dir: str,
    validation_dump_dir: str,
    test_dump_dir: str,
) -> None:
    for label in label2content:
        current_label_files: list[str] = list(label2content[label])
        number_of_files: int = len(current_label_files)
    
        start: int = 0
        end: int = ceil(number_of_files * train_size)
        train_files: list[str] = current_label_files[start:end]
    
        pathlib.Path(f'{train_dump_dir}/{label}/{additional_label}').mkdir(
            parents=True,
            exist_ok=True,
        )
    
        for filename in tqdm(train_files):
            for index, waveform in enumerate(label2content[label][filename]):
                torchaudio.save(
                    f'{train_dump_dir}/{label}/{additional_label}/{index}_{filename}.wav',
                    waveform,
                    sample_rate = sample_rate,
                    channels_first = True,
                    format='wav',
                )

        del train_files
    
        start = end
        end = start + ceil(number_of_files * validation_size)
        validation_files: list[str] = current_label_files[start:end]
    
        pathlib.Path(f'{validation_dump_dir}/{label}/{additional_label}').mkdir(
            parents=True,
            exist_ok=True,
        )
    
        for filename in tqdm(validation_files):
            for index, waveform in enumerate(label2content[label][filename]):
                torchaudio.save(
                    f'{validation_dump_dir}/{label}/{additional_label}/{index}_{filename}.wav',
                    waveform,
                    sample_rate = sample_rate,
                    channels_first = True,
                    format='wav',
                )

        del validation_files
    
        start = end
        test_files: list[str] = current_label_files[start:]
    
        pathlib.Path(f'{test_dump_dir}/{label}/{additional_label}').mkdir(
            parents=True,
            exist_ok=True,
        )
    
        for filename in tqdm(test_files):
            for index, waveform in enumerate(label2content[label][filename]):
                torchaudio.save(
                    f'{test_dump_dir}/{label}/{additional_label}/{index}_{filename}.wav',
                    waveform,
                    sample_rate = sample_rate,
                    channels_first = True,
                    format='wav',
                )


def preprocess_dataset_from_jose(
    all_files: list[str],
    sample_rate: int,
    chunk_size: int,
) -> dict[str, dict[str, list[torch.Tensor]]]:
    label2content: dict[str, dict[str, list[torch.Tensor]]] = {}

    for file in tqdm(all_files):
        label, filename = file.split('/')[-2:]
        filename = filename.split('.')[0]
    
        arr, current_sample_rate = torchaudio.load(file)
        if current_sample_rate != sample_rate:
            arr = torchaudio.functional.resample(
                arr,
                orig_freq=current_sample_rate,
                new_freq=sample_rate,
            )
        arr = torch.mean(
            arr,
            dim=0,
            keepdim=True,
        )
        
        chunks: tuple[torch.Tensor, ...] = torch.split(
            arr,
            chunk_size * sample_rate,
            dim=1,
        )
    
        for chunk in chunks:
            if chunk.shape[1] > sample_rate:
                if label not in label2content:
                    label2content[label] = {}
                if filename not in label2content[label]:
                    label2content[label][filename] = []
                label2content[label][filename].append(chunk)

    return label2content


def preprocess_nsu_dogs_and_weather_dataset(
    metainformation: pd.DataFrame,
    path_to_data_dir: str,
    dog_subclasses: list[str],
    sample_rate: int,
    chunk_size: int,
) -> dict[str, dict[str, list[torch.Tensor]]]:
    label2content: dict[str, dict[str, list[torch.Tensor]]] = {}

    for i in tqdm(metainformation.index):
        file_path: str = metainformation.loc[i].path
        filename: str = file_path.split('/')[-1].split('.')[0]
        label: str = 'other_animal' if metainformation.loc[i].sub_class in dog_subclasses else 'no_animal'
    
        arr, current_sample_rate = torchaudio.load(f'{path_to_data_dir}/{file_path}')
        if current_sample_rate != sample_rate:
            arr = torchaudio.functional.resample(
                arr,
                orig_freq=current_sample_rate,
                new_freq=sample_rate,
            )
        arr = torch.mean(
            arr,
            dim=0,
            keepdim=True,
        )

        chunks: tuple[torch.Tensor, ...] = torch.split(
            arr,
            chunk_size * sample_rate,
            dim=1,
        )
    
        for chunk in chunks:
            if chunk.shape[1] > sample_rate:
                if label not in label2content:
                    label2content[label] = {}
                if filename not in label2content[label]:
                    label2content[label][filename] = []
                label2content[label][filename].append(chunk)

    return label2content


def preprocess_nsu_wolfs_dogs_and_other_dataset(
    all_files: list[str],
    negative_markup: list[dict[str, tp.Any]],
    sample_rate: int,
    chunk_size: int,
) -> dict[str, dict[str, list[torch.Tensor]]]:
    label2content: dict[str, dict[str, list[torch.Tensor]]] = {}

    for file in tqdm(all_files):
        label: str = file.split('/')[-2]
        filename: str = file.split('/')[-1].split('.')[0]
    
        arr, current_sample_rate = torchaudio.load(file)
        if current_sample_rate != sample_rate:
            arr = torchaudio.functional.resample(
                arr,
                orig_freq=current_sample_rate,
                new_freq=sample_rate,
            )
        arr = torch.mean(
            arr,
            dim=0,
            keepdim=True,
        )
    
        chunks: tuple[torch.Tensor, ...] = torch.split(
            arr,
            chunk_size * sample_rate,
            dim=1,
        )
    
        if label == 'wolf':
            for chunk in chunks:
                if chunk.shape[1] > sample_rate:
                    if label not in label2content:
                        label2content[label] = {}
                    if filename not in label2content[label]:
                        label2content[label][filename] = []
                    label2content[label][filename].append(chunk)
        elif label == 'dog':
            global_label: str = 'other_animal'
            for chunk in chunks:
                if chunk.shape[1] > sample_rate:
                    if label not in label2content:
                        label2content[global_label] = {}
                    if filename not in label2content[global_label]:
                        label2content[global_label][filename] = []
                    label2content[global_label][filename].append(chunk)
        elif label == 'negative':
            for markup in negative_markup:
                if f'{filename}.wav' == markup['file_name']:
                    markup_label: str = markup['result']['type_speaker']
                    if markup_label == 'other_animal':
                        for chunk in chunks:
                            if chunk.shape[1] > sample_rate:
                                if markup_label not in label2content:
                                    label2content[markup_label] = {}
                                if filename not in label2content[markup_label]:
                                    label2content[markup_label][filename] = []
                                label2content[markup_label][filename].append(chunk)
                    elif markup_label in ['no_animal', 'insect']:
                        global_label = 'no_animal'
                        for chunk in chunks:
                            if chunk.shape[1] > sample_rate:
                                if global_label not in label2content:
                                    label2content[global_label] = {}
                                if filename not in label2content[global_label]:
                                    label2content[global_label][filename] = []
                                label2content[global_label][filename].append(chunk)
                    break

    return label2content


def preprocess_birdclef_dataset(
    all_files: list[str],
    sample_rate: int,
    chunk_size: int,
) -> dict[str, dict[str, list[torch.Tensor]]]:
    label2content: dict[str, dict[str, list[torch.Tensor]]] = {}

    for file in tqdm(all_files):
        filename: str = file.split('/')[-1].split('.')[0]
        label: str = 'other_animal'
    
        arr, current_sample_rate = torchaudio.load(file)
        if current_sample_rate != sample_rate:
            arr = torchaudio.functional.resample(
                arr,
                orig_freq=current_sample_rate,
                new_freq=sample_rate,
            )
        arr = torch.mean(
            arr,
            dim=0,
            keepdim=True,
        )
    
        chunks: tuple[torch.Tensor, ...] = torch.split(
            arr,
            chunk_size * sample_rate,
            dim=1,
        )
    
        for chunk in chunks:
            if chunk.shape[1] > sample_rate:
                if label not in label2content:
                    label2content[label] = {}
                if filename not in label2content[label]:
                    label2content[label][filename] = []
                label2content[label][filename].append(chunk)

    return label2content
