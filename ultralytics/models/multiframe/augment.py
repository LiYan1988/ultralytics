from copy import deepcopy

import numpy as np

from ultralytics.data.augment import Compose, LetterBox, Format


class MultiFrameCompose(Compose):
    """
    A class for composing multi-frame image transformations.

    Consider the following transformations:
    - LetterBox
    - Format
    """
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, data):
        for t in self.transforms:
            if isinstance(t, LetterBox):
                for i, img in enumerate(data['imgs']):
                    data['imgs'][i] = t(image=img)
            elif isinstance(t, Format):
                # Update data['img'] with images in data['imgs']
                # because we assume they are LetterBox transformed
                data['img'] = np.concatenate(data['imgs'], axis=-1)
                data = t(data)
        return data
