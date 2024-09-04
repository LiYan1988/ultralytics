# Multi-Frame Dataset
# Concatenate consecutive frames of a video into one sample

import contextlib
import json
from collections import defaultdict
from copy import deepcopy
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
import os
import math
import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset
import pandas as pd

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.torch_utils import TORCHVISION_0_18
from ultralytics.data.dataset import DATASET_CACHE_VERSION

from ultralytics.data.augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from ultralytics.data.base import BaseDataset
from ultralytics.data.utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label, exif_size,
)
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils.torch_utils import de_parallel

from .augment import MultiFrameCompose, MultiFrameLetterBox, multiframe_v8_transforms


class MultiFrameDataset(YOLODataset):
    """
    Dataset class for multi-frame keypoint detection.
    Multiple consecutive frames in a video are in one sample for action detection and object tracking.
    Assumes that all frames in a video is in the same directory or train/val/test split.
    Otherwise, we will not have consecutive frames.

    Args:
        data (dict): A dataset YAML directory.
        n_frames (int): Number of consecutive frames in a video to put into one sample.

    Returns:
        (torch.utils.data.Dataset): A dataset for multi-frame keypoint detection.
    """

    def __init__(self, *args, data, n_frames=5, file_sep='-', **kwargs):
        """
        Initializes a dataset object for keypoint detection tasks.
        """
        self.n_frames = n_frames
        # seperator between video name and frame index,
        #   e.g., '-' in 20240613_164634-860.txt or 20240613_164634-860.png
        self.file_sep = file_sep
        super().__init__(*args, data=data, **kwargs)

    def get_labels(self, ):
        """Get labels from txt files"""
        self.label_files = [str(_) for _ in (Path(self.img_path) / 'labels').rglob('*.txt')]
        # Do we need cache?
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} samples, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        # Note: self.im_files is not a list of frames, but a list of frame lists
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        if (self.fraction < 1) and (round(len(labels) * self.fraction) > 1):
            labels = labels[: round(len(labels) * self.fraction)]
        else:
            labels = labels[:2] # train and val may have problem when batch size equals 1
        return labels

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Overrides ultralytics.data.dataset.YOLODataset.cache_labels
        Even though the method name is `cache_labels`, it actually also creates a dictionary containing
        meta information of labels.
        The dictionary contains the following keys: labels, hash', 'results', 'msgs', 'version'
        """
        x = {"labels": []}
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )

        # Verify images
        desc = f"{self.prefix}Scanning images in {path.parent / 'images'}..."
        pbar = TQDM(enumerate(self.im_files), desc=desc, total=len(self.im_files))
        # pd.DataFrame containing meta info of images
        image_meta = pd.DataFrame(
            columns=['im_file', 'verified', 'shape', 'msg', 'video_name', 'frame_idx']
        )
        nc = 0
        msgs = []
        for idx, im_file in pbar:
            im_file, shape, nc_m, msg = self.verify_image(im_file, self.prefix)
            nc += nc_m
            image_meta.loc[idx, 'im_file'] = im_file
            image_meta.loc[idx, 'verified'] = True if nc_m == 0 else False
            image_meta.loc[idx, 'shape'] = shape
            image_meta.loc[idx, 'msg'] = msg
            if im_file is not None:
                video_name, frame_idx = Path(im_file).stem.rsplit(self.file_sep, maxsplit=1)
            else:
                video_name, frame_idx = None, None
            image_meta.loc[idx, 'video_name'] = video_name
            image_meta.loc[idx, 'frame_idx'] = int(frame_idx)
            pbar.desc = f"{desc}, {nc} corrupt."
            if msg:
                msgs.append(msg)
        pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        image_format = Path(im_file).suffix.split('.')[-1]

        # Verify labels
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        desc = f"{self.prefix}Scanning labels in {path.parent / 'labels'}..."
        pbar = TQDM(enumerate(self.label_files), desc=desc, total=len(self.label_files))
        # pd.DataFrame containing meta info of labels
        # label_meta = pd.DataFrame(
        #     columns=['label_file', 'verified', 'msg', 'im_files']
        # )
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        for idx, label_file in pbar:
            (
                lb_file, im_files, lb, keypoints,
                nm_f, nf_f, ne_f, nc_f, msg
            ) = self.verify_label(
                label_file,
                self.prefix,
                nkpt,
                ndim,
                len(self.data['names']),
                image_meta,
                self.n_frames,
                self.file_sep
            )
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f
            # Add label information to the output dictionary
            # Note: ultralytics.data.base.BaseDataset.__init__ requires 'im_file' as string,
            #   so we have to add another field 'im_files' containing all frames, and use 'im_file' for the last frame.
            x['labels'].append(
                {
                    'im_file': im_files[-1], # The last frame in the label
                    'im_files': im_files, # all frames in the label
                    'video_name': video_name,
                    'shape': shape, # assumes all images are of same shape
                    'cls': lb[:, 0:1],
                    'bboxes': lb[:, 1:],
                    'segments': [],
                    'keypoints': keypoints,
                    'normalized': True,
                    'bbox_format': 'xywh'
                }
            )
            if msg:
                msgs.append(msg)
            pbar.desc = f'{desc} {nf} labels, {nm + ne} backgrounds, {nc} corrupt'
        pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    @staticmethod
    def verify_image(im_file, prefix):
        """Verify one image."""
        msg = ""
        nc = 0 # number of corrupted image
        try:
            # Verify images
            im = Image.open(im_file)
            im.verify() # PIL verify
            shape = exif_size(im) # image size
            shape = (shape[1], shape[0]) # hw
            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} < 10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
            if im.format.lower() in {"jpg", "jpeg"}:
                with open(im_file, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                        msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
            return im_file, shape, nc, msg
        except Exception as e:
            nc = 1
            msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
            return None, None, nc, msg

    @staticmethod
    def verify_label(lb_file, prefix, nkpt, ndim, num_cls, image_meta, n_frames, file_sep):
        nm, nf, ne, nc, msg, keypoints = 0, 0, 0, 0, "", None
        try:
            if os.path.isfile(lb_file):
                nf = 1  # label found
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    lb = np.array(lb, dtype=np.float32)
                nl = len(lb)
                if nl:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                    assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                    assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                    # All labels
                    max_cls = lb[:, 0].max()  # max label count
                    assert max_cls <= num_cls, (
                        f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                        f"Possible class labels are 0-{num_cls - 1}"
                    )
                    _, i = np.unique(lb, axis=0, return_index=True)
                    if len(i) < nl:  # duplicate row check
                        lb = lb[i]  # remove duplicates
                        msg = f"{prefix}WARNING ⚠️ {lb_file}: {nl - len(i)} duplicate labels removed"
                else:
                    ne = 1  # label empty
                    lb = np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32)
            else:
                nm = 1  # label missing
                lb = np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32)
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
            lb = lb[:, :5]

            # Verify images in the label, even for background samples
            image_format = Path(image_meta.loc[0, 'im_file']).suffix.rsplit('.', 1)[1]
            video_name = Path(lb_file).stem.rsplit(file_sep, 1)[0]
            last_frame_idx = int(Path(lb_file).stem.rsplit(file_sep, 1)[1])
            img_dir = Path(lb_file).parents[1] / 'images'
            im_files = []
            for i in range(n_frames):
                im_name_ = f'{video_name}{file_sep}{last_frame_idx - n_frames + 1 + i}.{image_format}'
                im_file_ = img_dir / im_name_
                assert im_file_.is_file(), f'{im_file_} does not exist'
                im_files.append(str(im_file_))
            if image_meta.loc[image_meta['im_file'].isin(im_files), 'verified'].all():
                return lb_file, im_files, lb, keypoints, nm, nf, ne, nc, msg
            else:
                corrupted_images = image_meta.loc[
                    image_meta['im_file'].isin(im_files) & (image_meta['verified'] == False),
                    'im_file'
                ].tolist()
                raise Exception(f'Corrupted images are: {corrupted_images}')
        except Exception as e:
            nc = 1
            msg = f"{prefix}WARNING ⚠️ {lb_file}: ignoring corrupt image/label: {e}"
            return [None, None, None, None, nm, nf, ne, nc, msg]

    @staticmethod
    def load_single_image(img_file, imgsz, rect_mode=True):
        """
        Load image in img_file. If rect_mode is True resize to maintain aspect ratio,
        otherwise stretch image to square imgsz.
        Returns (im, original hw, resized hw).
        """
        im = cv2.imread(str(img_file))
        h0, w0 = im.shape[:2] # original hw
        if rect_mode:
            r = imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                w, h = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == imgsz):  # resize by stretching image to square imgsz
            im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]

    def load_image(self, i, rect_mode=True):
        """
        Load image according to labels' im_files field.
        """
        label = self.labels[i]
        img_files = label['im_files']
        imgs, ori_shape, resized_shape = [None] * self.n_frames, [None] * self.n_frames, [None] * self.n_frames
        for j, f in enumerate(img_files):
            imgs[j], ori_shape[j], resized_shape[j] = self.load_single_image(f, self.imgsz, rect_mode)
        return imgs, ori_shape[0], resized_shape[0]

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        # label["img"] contains each individual frame, and label["imgs"] contain the concatenated multiple frames
        label["imgs"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["img"] = np.concatenate(label["imgs"], axis=-1)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def build_transforms(self, hyp=None):
        """
        Build and appends transforms to the list for multi-frame dataset.

        Consider the following transformations at the moment:
        - LetterBox
        - Format
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = multiframe_v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = MultiFrameCompose([MultiFrameLetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        # No need to apply fraction here as we will do it in self.get_labels
        # if self.fraction < 1:
        #     im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
        return im_files
