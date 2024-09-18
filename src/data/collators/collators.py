import torch
import torch.nn.functional as F


def spectrogram_collate_function(
    batches: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    spectrograms: list[torch.Tensor] = []
    classes: list[int] = []

    max_length: int = 0
    for current_spectrogram, _ in batches:
        max_length = max(
            max_length,
            current_spectrogram.shape[-1],
        )

    for current_spectrogram, current_class in batches:
        spectrograms.append(
            F.pad(
                current_spectrogram,
                pad=(
                    2,
                    max_length - current_spectrogram.shape[-1],
                ),
                mode='constant',
                value=0,
            ),
        )
        classes.append(current_class)

    return (
        torch.stack(spectrograms),
        torch.LongTensor(classes),
    )


def wav_collate_function(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    waveforms: list[torch.Tensor] = []
    classes: list[int] = []

    for current_waveform, current_class in batch:
        waveforms.append(current_waveform.T)
        classes.append(current_class)

    return (
        torch.nn.utils.rnn.pad_sequence(
            waveforms,
            batch_first=False,
            padding_value=0,
        ).squeeze(-1).T,
        torch.LongTensor(classes),
    )


def speechbrain_collate_function(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    waveform_batch, labels_batch = wav_collate_function(batch)

    return waveform_batch.unsqueeze(-1), labels_batch
