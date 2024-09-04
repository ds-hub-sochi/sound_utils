import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.metrics.metrics import target_class_precision, target_class_recall


@torch.inference_mode()
def get_targets_and_predicitons(
    model,
    dataloader,
    device,
) -> tuple[list[float], list[float]]:
    val_predictions: list[float] = []
    val_targets: list[float] = []

    for batch, targets in tqdm(dataloader):        
        batch: torch.Tensor = batch.to(device)
        targets: torch.Tensor = targets.to(device)
        predictions: torch.Tensor = model(batch)
                        
        val_predictions.extend(predictions.cpu().numpy().argmax(axis=1))
        val_targets.extend(targets.cpu().numpy())

    return (
        val_predictions,
        val_targets,
    )

@torch.inference_mode()
def get_precision_recall_threshold(
    model,
    dataloader,
    device,
) -> tuple[list[float], list[float], list[float]]:
    val_probabilities: list[float] = []
    val_targets: list[float] = []

    for batch, targets in tqdm(dataloader):        
        batch: torch.Tensor = batch.to(device)
        predictions: torch.Tensor = model(batch)

        val_targets.extend(targets.cpu().numpy())
        val_probabilities.extend([round(value, 2) for value in F.softmax(predictions, dim=1)[:, 0].cpu().numpy()])

    precision_scores: list[float] = []
    recall_scores: list[float] = []

    thresholds: list[float] = sorted(list(set(val_probabilities)))

    for threshold in thresholds:
        predicted_classes = np.where(np.array(val_probabilities) > threshold, 0, 1)
        precision_scores.append(target_class_precision(predicted_classes, val_targets, 0))
        recall_scores.append(target_class_recall(predicted_classes, val_targets, 0))

    return (
        precision_scores,
        recall_scores,
        thresholds,
    )
