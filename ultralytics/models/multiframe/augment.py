import random
from copy import deepcopy
import cv2

import numpy as np

from ultralytics.data.augment import (
    Compose,
    LetterBox,
    Format,
    Mosaic,
    RandomPerspective,
    MixUp,
    Albumentations,
    RandomHSV,
    RandomFlip
)
from ultralytics.utils import LOGGER


"""
Apply augmentations to multi-frame data. There are two types of augmentations:
1. Channel-agnostic transformation: 
    - we first apply in data['img'] for all frames, 
    - then update each frame in data['imgs']
    - transformations in this type are:
        - Mosaic
        - RandomPerspective
        - Mixup
        - RandomFlip
2. Channel dependent transformation, where input data should be a 3-channel image
    - we first apply the same transformation on each frame in data['img']
    - then concatenate all frames into data['imgs'] 
    - transformations in this type are:
        - RandomHSV
        - LetterBox
"""


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

    def __call__(self, data=None, images=None):
        """
        For multi-frame task, apply letterboxing to each image in a multi-frame series by resizing the image
        while maintaining its aspect ratio and adding padding to fit the new shape. It also updates any
        associated labels accordingly during training and validation.

        To perform LetterBox for multi-frame images, we first transform each individual frame in data['imgs'] or images.
        Then concatenate transformed frames into data['img'] or output the np.ndarray (if `images` is provided).

        Args:
            data (dict | None): A dictionary containing image data and associated labels during training and validation.
                It can also be None during inference.
            images (list | None): A list containing a multi-frame series of images. This argument is used
                in inference.
        """
        if images is None:
            for i, img in enumerate(data['imgs']):
                data['imgs'][i], left, top, ratio, dw, dh, new_shape = self._transform_single_image(img)

            data["ratio_pad"] = (data["ratio_pad"], (left, top))  # for evaluation

            data = self._update_labels(data, ratio, dw, dh)
            data["resized_shape"] = new_shape

            # Update final image by concatenate transformed images
            data['img'] = np.concatenate(data['imgs'], axis=-1)
            return data
        else:
            ims = [None] * len(images)
            for i, img in enumerate(images):
                ims[i], *_ = self._transform_single_image(img)
            return np.concatenate(ims, axis=-1)


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
        data_ = super().__call__(labels=data)
        return update_imgs_from_img(data, data_)


class MultiFrameRandomPerspective(RandomPerspective):

    def __call__(self, data):
        data_ = super().__call__(labels=data)
        return update_imgs_from_img(data, data_)


class MultiFrameMixup(MixUp):

    def __call__(self, data):
        data_ = super().__call__(labels=data)
        return update_imgs_from_img(data, data_)


class MultiFrameRandomHSV(RandomHSV):
    """
    Apply RandomHSV on multi-frame image.

    Initializes random gains before transformation to make sure
        the same parameters are applied to all images in the multi-frame series.
    """

    def _random_hsv(self, dtype):
        """Generate random gains and create look up tables."""
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        return lut_hue, lut_sat, lut_val

    def _apply(self, img, lut_hue, lut_sat, lut_val):
        """Apply HSV transform on a single frame image."""
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        im_hsv = cv2.merge(
            (
                cv2.LUT(hue, lut_hue),
                cv2.LUT(sat, lut_sat),
                cv2.LUT(val, lut_val)
            )
        )
        # We do not manipulate on the original image but return a copy.
        # Because in-place change may require a contiguous array.
        return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)

    def __call__(self, data):
        """
        Applies the same HSV transform on each individual frame and concatenate frames.
        """
        lut_hue, lut_sat, lut_val = self._random_hsv(data['img'].dtype)
        for i, img in enumerate(data['imgs']):
            data['imgs'][i] = self._apply(img, lut_hue, lut_sat, lut_val)
        data['img'] = np.concatenate(data['imgs'], axis=-1)
        return data


# This is not needed because RandomFlip is channel agnostic.
# But still keep it here for future references in case we need to implement other multi-frame augmentations.
class MultiFrameRandomFlip(RandomFlip):
    """
    Apply RandomFlip on multi-frame image.
    """

    def _prepare_transform(self, data):
        """
        Prepare random number and image shape for one sample, i.e., a series of multiple frames.
        """
        img = data['img']
        instances = data.get('instances')
        instances.convert_bbox(format='xywh')
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w
        return instances, w, h

    def _apply_on_image(self, img):
        """Apply flip on a single image."""
        if self.direction == "vertical":
            img = np.flipud(img)
        if self.direction == "horizontal":
            img = np.fliplr(img)
        return np.ascontiguousarray(img)

    def _apply_on_instances(self, instances, w, h):
        """Apply flip on ground truths."""
        if self.direction == "vertical":
            instances.flipud(h)
        if self.direction == "horizontal":
            instances.fliplr(w)
            # For keypoints
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        return instances

    # Will not use this __call__ method
    # def __call__(self, data):
    #     # Generate a random number
    #     r = random.random()
    #     if r >= self.p:
    #         # Directly return the original data
    #         return data
    #
    #     instances, w, h = self._prepare_transform(data)
    #     # Apply the same flip on all frames
    #     for i, img in enumerate(data['imgs']):
    #         data['imgs'][i] = self._apply_on_image(img)
    #     data['img'] = np.ascontiguousarray(
    #         np.concatenate(data['imgs'], axis=-1)
    #     )
    #     # Apply corresponding transformation on labels
    #     data['instances'] = self._apply_on_instances(instances, w, h)
    #     return data

    def __call__(self, data):
        data_ = super().__call__(labels=data)
        return update_imgs_from_img(data, data_)


class MultiFrameFormat(Format):
    """
    Overrides ultralytics.data.augment.Format for the multiframe task.

    Ensure correct frame and channel orders:
    - Frames are from old to new, first the oldest, last the newest.
    - Channel order of each frame is RGB. Convert BGR to RGB for each individual frame and then concatenate frames.
    """

    def __call__(self, data):
        """
        Ensure output is in the correct channel order.

        The desired output channel order is R1G1B1R2G2B2...RnGnBn, where n is the number of frames.
        The corresponding desired input channel should be BnGnRn...B1G1R1
        The actual input channel order is B1G1R1B2G2R2...BnGnRn.
        So we just need to first reverse the frame order to get the desired channel order.
        Then we can call the original Format transform.
        """
        n_frames = data['img'].shape[-1] // 3
        imgs = [data['img'][..., i * 3 : (i + 1) * 3] for i in reversed(range(n_frames))]
        data['img'] = np.concatenate(imgs, axis=-1)
        data = super().__call__(data)
        return data


def update_imgs_from_img(data_orig, data_aug):
    """
    Updates individual frame from concatenated multi-frame image.
    Also copies other properties from data_aug to data_orig.
    """
    data = data_orig | data_aug
    for i in range(len(data['imgs'])):
        data['imgs'][i] = data['img'][..., i * 3: (i + 1) * 3]
    # Pop unwanted properties
    data.pop('mix_labels', None)
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
    pre_transform = Compose(
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

    return Compose(
        [
            pre_transform,
            MultiFrameMixup(dataset, pre_transform=pre_transform, p=hyp.mixup),
            # Albumentations(p=1.0), # Albumentations augment is disabled at the moment
            MultiFrameRandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            MultiFrameRandomFlip(direction="vertical", p=hyp.flipud),
            MultiFrameRandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms