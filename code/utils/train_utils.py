import logging
from sklearn.metrics import accuracy_score
from datetime import datetime
import os
import json

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value
    https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def as_minutes(s):
    """Convert seconds to minutes:seconds format."""
    m = int(s / 60)
    s -= m * 60
    return f"{m:02d}:{int(s):02d}"


def get_lr(optimizer):
    """Get the current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def save_training_log(config, log_data, is_best=False):
    log_dir = os.path.join(config.outputs.model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.json")

    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)

    if is_best:
        best_log_file = os.path.join(log_dir, "best_model_log.json")
        with open(best_log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    return log_file


# ================================================


def compute_metrics(predictions, truths):
    """
    ACCURACY SCORE
    """

    assert len(predictions) == len(truths)

    return {
        "accuracy": round(accuracy_score(truths, predictions), 4),
    }
