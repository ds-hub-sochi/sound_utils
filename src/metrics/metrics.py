def target_class_precision(
    real_classes: list[int],
    predicted_classes: list[int],
    target_class: int,
) -> float:
    true_positive_cases: int = 0
    false_positive_cases: int = 0

    eps: float = 1e-6
    for predicted_class, real_class in zip(predicted_classes, real_classes):
        if (predicted_class == target_class) and (real_class == target_class):
            true_positive_cases += 1
        if (predicted_class == target_class) and (real_class != target_class):
            false_positive_cases += 1

    return true_positive_cases/(true_positive_cases + false_positive_cases + eps)


def target_class_recall(
    real_classes: list[int],
    predicted_classes: list[int],
    target_class: int,
) -> float:
    true_positive_cases: int = 0
    false_negative_cases: int = 0

    eps: float = 1e-6
    for predicted_class, real_class in zip(predicted_classes, real_classes):
        if (predicted_class == target_class) and (real_class == target_class):
            true_positive_cases += 1
        if (predicted_class != target_class) and (real_class == target_class):
            false_negative_cases += 1

    return true_positive_cases/(true_positive_cases + false_negative_cases + eps)