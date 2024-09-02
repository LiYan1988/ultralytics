from copy import deepcopy
import cv2

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
            if isinstance(t, MultiFrameLetterBox):
                data = t(data)
                # for i, img in enumerate(data['imgs']):
                #     data['imgs'][i] = t(image=img)
            elif isinstance(t, Format):
                # Update data['img'] with images in data['imgs']
                # because we assume they are LetterBox transformed
                data['img'] = np.concatenate(data['imgs'], axis=-1)
                data = t(data)
        return data


class MultiFrameLetterBox(LetterBox):
    """
    Overrides __call__ method of LetterBox to return ratio_pad when labels=None in input argument.
    """
    def _transform_single_image(self, img):
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        return img, left, top, ratio, dw, dh, new_shape

    def __call__(self, labels):
        for i, img in enumerate(labels['imgs']):
            labels['imgs'][i], left, top, ratio, dw, dh, new_shape = self._transform_single_image(img)

        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            # labels["img"] = img # labels['imgs'] is updated instead,
            labels["resized_shape"] = new_shape
        return labels
