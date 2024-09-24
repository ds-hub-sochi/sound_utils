import pathlib

import numpy as np
import torch
from sklearn.metrics import fbeta_score
from tqdm import tqdm

from src.metrics.metrics import target_class_precision, target_class_recall


def train_step(  # pylint: disable=[too-many-positional-arguments]
    model,
    optimizer,
    criterion,
    train_dataloader,
    epoch: int,
    device,
    scaler,
) -> dict[str, float]:
    model.train()

    train_loss: list[float] = []
    train_predictions: list[int] = []
    train_targets: list[int] = []

    if scaler is not None:
        with torch.cuda.amp.autocast():
            for batch, targets in tqdm(train_dataloader, desc=f'Epoch: {epoch}'):
                batch = batch.to(device)
                targets = targets.to(device)

                predictions = model(batch)

                optimizer.zero_grad()
                loss = criterion(predictions, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss.append(loss.item())
                train_predictions.extend(predictions.cpu().detach().numpy().argmax(axis=1))
                train_targets.extend(targets.cpu().detach().numpy())
    else:
        for batch, targets in tqdm(train_dataloader, desc=f'Epoch: {epoch}'):
            batch = batch.to(device)
            targets = targets.to(device)

            predictions = model(batch)

            optimizer.zero_grad()
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_predictions.extend(predictions.cpu().detach().numpy().argmax(axis=1))
            train_targets.extend(targets.cpu().detach().numpy())

    train_fbeta: float = fbeta_score(train_targets, train_predictions, average='macro', beta=2)
    train_precision: float = target_class_precision(train_targets, train_predictions, 0)
    train_recall: float = target_class_recall(train_targets, train_predictions, 0)

    return {
        'loss': np.mean(train_loss),
        'f2-macro': train_fbeta,
        'target recall': train_recall,
        'target precision': train_precision,
    }


@torch.inference_mode()
def eval_step(
    model,
    criterion,
    val_dataloader,
    epoch: int,
    device,
) -> dict[str, float]:
    model.eval()

    val_predictions = []
    val_targets = []
    val_loss = []

    for batch, targets in tqdm(val_dataloader, desc=f'Epoch: {epoch}'):
        batch: torch.Tensor = batch.to(device)
        targets: torch.Tensor = targets.to(device)
        predictions: torch.Tensor = model(batch)
        loss: torch.Tensor = criterion(predictions, targets)

        val_loss.append(loss.item())
        val_predictions.extend(predictions.cpu().numpy().argmax(axis=1))
        val_targets.extend(targets.cpu().numpy())

    val_fbeta: float = fbeta_score(val_targets, val_predictions, average='macro', beta=2)
    val_precision: float = target_class_precision(val_targets, val_predictions, 0)
    val_recall: float = target_class_recall(val_targets, val_predictions, 0)

    return {
        'loss': np.mean(val_loss),
        'f2-macro': val_fbeta,
        'target recall': val_recall,
        'target precision': val_precision,
    }


def save_model(
    model: torch.nn.Module,
    checkpoint_dir: str,
    prefix: str,
    name: str,
) -> None:
    pathlib.Path(f'{checkpoint_dir}/{prefix}').mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(
        model,
        'module',
    ):
        torch.save(
            model.module.state_dict(),
            f'{checkpoint_dir}/{prefix}/{name}.pth',
        )
    else:
        torch.save(
            model.state_dict(),
            f'{checkpoint_dir}/{prefix}/{name}.pth',
        )


def train(  # pylint: disable=[too-many-arguments,too-many-positional-arguments]
    model,
    optimizer,
    criterion,
    train_dataloader,
    val_dataloader,
    n_epochs,
    checkpoint_dir: str,
    prefix: str,
    device,
    scaler=None,
    scheduler=None,
) -> dict[str, dict[str, float]]:
    pathlib.Path(f'{checkpoint_dir}/{prefix}').mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_storage: dict[str, dict[str, float]] = {}

    keys: list[str] = [
        'loss',
        'f2-macro',
        'target precision',
        'target recall',
    ]
    for key in keys:
        metric_storage[key] = {
            'train': [],
            'val': [],
        }

    max_fbeta: float = 0.0

    for epoch in range(n_epochs):
        train_results: dict[str, float] = train_step(
            model,
            optimizer,
            criterion,
            train_dataloader,
            epoch,
            device,
            scaler,
        )

        for key in keys:
            print(f'Train {key}:', train_results[key], end='\n')
            metric_storage[key]['train'].append(train_results[key])

        eval_results: dict[str, float] = eval_step(
            model,
            criterion,
            val_dataloader,
            epoch,
            device,
        )

        for key in keys:
            print(f'Eval {key}:', eval_results[key], end='\n')
            metric_storage[key]['val'].append(eval_results[key])

        if eval_results['f2-macro'] > max_fbeta:
            max_fbeta = eval_results['f2-macro']
            save_model(
                model,
                checkpoint_dir,
                prefix,
                'best_model',
            )
            print('Checkpoint updated', end='\n')

        if scheduler is not None:
            scheduler.step()

    save_model(
        model,
        checkpoint_dir,
        prefix,
        'last_model',
    )

    return metric_storage
