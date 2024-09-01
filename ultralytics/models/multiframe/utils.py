# Simple modifications to make multi-frame work
# Principles:
# - No over-engineering, no perfect, only problem-solving.

import glob
from pathlib import Path


def mf_get_label_files(img_path):
    """
    Get label file paths by search for txt files in the label directory.
    This is used for multi-frame where the image and label files do not have one-to-one mapping.
    """
    # img_path is actually the train path,
    # for example '/Users/li.yan/learn/table-tennis/data/datasets/ball-keypoint/20240813/train'
    label_dir = Path(img_path) / 'labels'
    return [str(_) for _ in label_dir.rglob('*.txt')]
