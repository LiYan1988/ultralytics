from copy import deepcopy
import cv2

import numpy as np

from ultralytics.data.augment import (
    Compose,
    LetterBox,
    Format,
    Mosaic,
    RandomPerspective,
    MixUp
)
from ultralytics.utils import LOGGER


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
            data = t(data)
            # TODO: refactor later
            # if isinstance(t, Format):
            #     # Update data['img'] with images in data['imgs']
            #     # because we assume they are LetterBox transformed
            #     data['img'] = np.concatenate(data['imgs'], axis=-1)
            #     data = t(data)
            # else:
            #     data = t(data)
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

    def __call__(self, data):
        """
        First perform LetterBox on each individual frame in data['imgs'].
        Then concatenate transformed frames into data['img'].
        """
        for i, img in enumerate(data['imgs']):
            data['imgs'][i], left, top, ratio, dw, dh, new_shape = self._transform_single_image(img)

        data["ratio_pad"] = (data["ratio_pad"], (left, top))  # for evaluation

        data = self._update_labels(data, ratio, dw, dh)
        data["resized_shape"] = new_shape

        # Update final image by concatenate transformed images
        data['img'] = np.concatenate(data['imgs'], axis=-1)
        return data


class MultiFrameMosaic(Mosaic):
    """
    Mosaic transformation for multi-frame images. Just disable the buffer in dataset
        and randomly choose from dataset to create a mosaic image.
    """
    def get_indexes(self, buffer=False):
        """
        Return a list of random indexes from the dataset for mosaic augmentation.

        The original method in Mosaic selects from a buffer.
        But we will directly select from dataset.
        """
        return super().get_indexes(buffer)

    def __call__(self, data):
        """
        First perform Mosaic transformation on data['img'], then update data['imgs'] one by one.
        """
        super().__call__(labels=data)
        return update_imgs_from_img(data)


class MultiFrameRandomPerspective(RandomPerspective):

    def __call__(self, data):
        super().__call__(labels=data)
        return update_imgs_from_img(data)


class MultiFrameMixup(MixUp):

    def __call__(self, data):
        super().__call__(labels=data)
        return update_imgs_from_img(data)


def update_imgs_from_img(data):
    """
    Update individual frame from concatenated multi-frame image.
    """
    # assert len(data['imgs']) * 3 == data['imgs'].shape[-1]
    for i in range(len(data['imgs'])):
        data['imgs'][i] = data['img'][..., i * 3: (i + 1) * 3]
    return data


def multiframe_v8_transforms(dataset, imgsz, hyp, stretch=False):
    """
    Overrides ultralytics.data.augment.v8_transforms for multiframe model.

    Applies a series of image transformations for YOLOv8 training.

    This function creates a composition of image augmentation techniques to prepare images for YOLOv8 training.
    It includes operations such as mosaic, copy-paste, random perspective, mixup, and various color adjustments.

    Args:
        dataset (Dataset): The dataset object containing image data and annotations.
        imgsz (int): The target image size for resizing.
        hyp (Dict): A dictionary of hyperparameters controlling various aspects of the transformations.
        stretch (bool): If True, applies stretching to the image. If False, uses LetterBox resizing.

    Returns:
        (Compose): A composition of image transformations to be applied to the dataset.

    Examples:
        >>> from ultralytics.data.dataset import YOLODataset
        >>> dataset = YOLODataset(img_path="path/to/images", imgsz=640)
        >>> hyp = {"mosaic": 1.0, "copy_paste": 0.5, "degrees": 10.0, "translate": 0.2, "scale": 0.9}
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data = transforms(dataset[0])
    """
    pre_transform = MultiFrameCompose(
        [
            MultiFrameMosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
            MultiFrameRandomPerspective(
                degrees=hyp.degrees,
                translate=hyp.translate,
                scale=hyp.scale,
                shear=hyp.shear,
                perspective=hyp.perspective,
                pre_transform=None if stretch else MultiFrameLetterBox(new_shape=(imgsz, imgsz)),
            ),
        ]
    )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    return MultiFrameCompose(
        [
            pre_transform,
            MultiFrameMixup(dataset, pre_transform=pre_transform, p=hyp.mixup),
            # Albumentations(p=1.0),
            # RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            # RandomFlip(direction="vertical", p=hyp.flipud),
            # RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms
