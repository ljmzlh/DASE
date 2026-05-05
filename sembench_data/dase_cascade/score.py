"""Common scoring helpers used across cascade evaluations."""
from typing import Iterable, Optional, Sequence


def f1_set(predicted: Iterable, gt: Iterable):
    """Set-retrieval (precision, recall, f1) over hashable items."""
    pred = set(predicted)
    truth = set(gt)
    if not truth:
        f1 = 1.0 if not pred else 0.0
        return 1.0 if not pred else 0.0, 1.0, f1
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def relative_error_score(predicted: float, gt: float) -> float:
    """Aggregation score: score = 1 / (1 + |pred - gt| / gt). Used by wildlife COUNT Qs."""
    if gt == 0:
        return 1.0 if predicted == 0 else 0.0
    return 1.0 / (1.0 + abs(predicted - gt) / gt)


def ari_score(pred_labels: Sequence, gt_labels: Sequence) -> float:
    """Adjusted Rand Index. Used by ecomm Q3 / Q12 brand/color clustering Qs."""
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(list(gt_labels), list(pred_labels)))
