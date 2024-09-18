import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.metrics.metrics import target_class_precision, target_class_recall


@torch.inference_mode()
def get_targets_and_predicitons(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
) -> tuple[list[float], list[float]]:
    device: torch.device = next(model.parameters()).device

    val_predictions: list[float] = []
    val_targets: list[float] = []

    for batch, targets in tqdm(dataloader):        
        batch: torch.Tensor = batch.to(device)
        predictions: torch.Tensor = model(batch)
                        
        val_predictions.extend(predictions.cpu().numpy().argmax(axis=1))
        val_targets.extend(targets.cpu().numpy())

    return (
        val_targets,
        val_predictions,
    )

@torch.inference_mode()
def get_precision_recall_threshold(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    target_class: int,
) -> tuple[list[float], list[float], list[float]]:
    device: torch.device = next(model.parameters()).device

    val_probabilities: list[float] = []
    val_targets: list[float] = []

    for batch, targets in tqdm(dataloader):        
        batch: torch.Tensor = batch.to(device)
        predictions: torch.Tensor = model(batch)

        val_targets.extend(targets.cpu().numpy())
        val_probabilities.extend([round(value, 2) for value in F.softmax(predictions, dim=1)[:, target_class].cpu().numpy()])

    precision_scores: list[float] = []
    recall_scores: list[float] = []

    thresholds: list[float] = sorted(list(set(val_probabilities)))

    other_class: int = 1 if not target_class else 0
    
    for threshold in thresholds:
        predicted_classes = np.where(
            np.array(val_probabilities) > threshold,
            target_class,
            other_class,
        )
        precision_scores.append(
            target_class_precision(
                real_classes=val_targets,
                predicted_classes=predicted_classes,
                target_class=target_class,
            )
        )
        recall_scores.append(
            target_class_recall(
                real_classes=val_targets,
                predicted_classes=predicted_classes,
                target_class=target_class,
            )
        )

    return (
        precision_scores,
        recall_scores,
        thresholds,
    )
