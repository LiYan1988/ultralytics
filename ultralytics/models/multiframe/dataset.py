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
import copy

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import ConcatDataset
import pandas as pd

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr, plotting
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
    verify_image_label,
    exif_size,
    FORMATS_HELP_MSG,
)
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes
from ultralytics.engine.results import Results

from .augment import MultiFrameLetterBox, multiframe_v8_transforms, MultiFrameFormat
from .utils import MultiFrameAnnotator


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

    def __init__(self, *args, data, n_frames, file_sep='-', **kwargs):
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
        # self.im_files = [lb["im_file"] for lb in labels]  # update im_files

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
        if self.fraction < 1:
            if round(len(labels) * self.fraction) > 1:
                labels = labels[: round(len(labels) * self.fraction)]
            else:
                # Only loads 2 labels if self.fraction is too small
                # This is for quick check in development
                # train and val may have problem when batch size equals 1, so at least 2 samples.
                labels = labels[:2]

        # Update self.ni and self.im_files.
        # In the original YOLO implementation:
        # - self.im_files equals to all images in the image directory, as implemented
        #     in ultralytics.data.base.BaseDataset.get_img_files.
        #     But we may not use all of them, for example if fraction < 1.
        # NOTE: The order of elements (image paths) in self.im_files is not aligned with those in video or self.labels.
        #   The original implementation indexes self.im_files like self.im_files[i] to get the path to the i-th image.
        #   This is not applicable in multiframe task. Because indexing from dataloader is referring to
        #   multiframe series instead of images.
        self.im_files = sorted(set([_ for l in labels for _ in l['im_files']]))

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

    def load_single_image(self, im_file, rect_mode=True):
        """
        Load image in img_file. If rect_mode is True resize to maintain aspect ratio,
        otherwise stretch image to square imgsz.

        The original implementation in ultralytics.data.base.BaseDataset.load_image
        loads image by its index in self.im_files. This is not the case in multiframe task.
        Correspondingly, self.ims, self.npy_files, self.im_hw0, self.im_hw should be dict with im_file as key.

        Returns (im, original hw, resized hw).
        """
        # Check cache, similar as the first part in ultralytics.data.base.BaseDataset.load_image.
        # With im_file as key to self.ims and self.npy_files
        im, fn = self.ims[im_file], self.npy_files[im_file]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(im_file)  # BGR
            else:  # read image
                im = cv2.imread(im_file)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {im_file}")

            h0, w0 = im.shape[:2] # original hw
            if rect_mode:
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Don't understand this part:
            # /opt/miniconda3/envs/table-tennis-analytics/lib/python3.12/site-packages/ultralytics/data/base.py:174
            self.ims[im_file], self.im_hw0[im_file], self.im_hw[im_file] = im, (h0, w0), im.shape[:2]

            return im, (h0, w0), im.shape[:2]

        return self.ims[im_file], self.im_hw0[im_file], self.im_hw[im_file]

    def cache_images(self):
        """
        Cache images to memory or disk.

        Overrides the following properties in ultralytics.data.base.BaseDataset:
        - self.ims,
        - self.im_hw0,
        - self.im_hw,
        - self.npy_files,
        """
        self.ims = {f: None for f in self.im_files}
        self.im_hw0 = {f: None for f in self.im_files}
        self.im_hw = {f: None for f in self.im_files}
        self.npy_files = {f: Path(f).with_suffix(".npy") for f in self.im_files}
        # - self.ni equals to the number of labels.
        #     But in multiframe task the number of labels does not equal to the number of images.
        self.ni = len(self.im_files)

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_single_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, self.im_files)
            pbar = TQDM(zip(self.im_files, results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, im_file):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[im_file] # f is a pathlib.Path object
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(im_file), allow_pickle=False)

    def load_image(self, i, rect_mode=True):
        """
        Loads image according to labels' im_files field.

        Loads the i-th multiframe series, where i = 0, 1, ..., N-n.
        The returned imgs is not an image, but a list of images.
        """
        label = self.labels[i]
        img_files = label['im_files']
        imgs, ori_shape, resized_shape = [None] * self.n_frames, [None] * self.n_frames, [None] * self.n_frames
        for j, f in enumerate(img_files):
            # The j-th image of the i-th series, with j = 0, 1, ..., n -1 (n=self.n_frames), and i = 0, 1, ..., N-n.
            imgs[j], ori_shape[j], resized_shape[j] = self.load_single_image(f)
        return imgs, ori_shape[0], resized_shape[0]

    def get_image_and_label(self, index):
        """
        Get and return label information from the dataset.

        Args:
            index: the index of multiframe series, 0, 1, ..., N-n.
        """
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
            transforms = Compose([MultiFrameLetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            MultiFrameFormat(
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


class MultiFrameVideoLoader(LoadImagesAndVideos):
    """
    Load videos for predict mode in the Multi-Frame model.
    """

    def __init__(self, path, n_frames, batch=1):
        # Initialize self.n_frames before initiliazing the parent class
        # because we need to use self.n_frames in self._new_video method.
        self.n_frames = n_frames
        super().__init__(path, batch=batch)
        self.frame_buffer = []
        self.source_type = SourceTypes(stream=False, screenshot=False, from_img=False, tensor=False)

    def _update_frame_buffer(self, im0):
        self.frame_buffer.append(im0)
        while len(self.frame_buffer) > self.n_frames:
            self.frame_buffer.pop(0)

    def _retrieve_multiframe_images(self, path):
        """Retrieve a series of frames from video."""
        if not self.cap or not self.cap.isOpened():
            self._new_video(path)

        while len(self.frame_buffer) < self.n_frames:
            success = self.cap.grab()
            if success:
                success, im0 = self.cap.retrieve()
                # No need to doublecheck image retrieval success.
                # Just directly append the newest frame to the end of the frame.
                self.frame_buffer.append(im0)
            else:
                # If retrieval is not successful, we are at the end of the video.
                break

        # It is not possible for success == True and len(self.frame_buffer) < self.n_frames
        # So if success == True, we always have len(self.frame_buffer) >= self.n_frames
        if success:
            # Ensure n_frames images in the buffer
            while len(self.frame_buffer) > self.n_frames:
                self.frame_buffer.pop(0)
            # If retrieval of frame is successful and the frame_buffer is full, output multi-frame series.
            # TODO: self.frame and self.frames should be modified to reflect the number of multi-frame series.
            self.frame += 1
            multi_frame_img = copy.deepcopy(self.frame_buffer)
            # Remove the oldest frame from the buffer
            self.frame_buffer.pop(0)
            info = f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: "
        else:
            # We are at the end of the video
            # NOTE: when clipping from a video, the original number of frames from cv2.CAP_PROP_FRAME_COUNT
            # is not accurate.
            # It maybe represents the frame index of the last frame,
            # instead of the total number of frames contained in the video clip.
            self.count += 1
            if self.cap:
                self.cap.release()
            if self.count < self.nf:
                self._new_video(self.files[self.count])
            # Clear self.frame_buffer
            self.frame_buffer = []
            return path, None, None
        return path, multi_frame_img, info

    def __next__(self):
        """
        Returns the next batch of multi-frame series from the video with their paths and metadata.
        """
        self.mode = 'video'

        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:
                if imgs:
                    return paths, imgs, info
                else:
                    raise StopIteration

            path = self.files[self.count]
            if not self.cap or not self.cap.isOpened():
                self._new_video(path)

            path_, img_, info_ = self._retrieve_multiframe_images(path)
            if img_ is not None:
                paths.append(path_)
                imgs.append(img_)
                info.append(info_)
        return paths, imgs, info

    def _new_video(self, path):
        """
        In the multi-frame model, create a new video capture object for the given path.
        Calculate the correct number of multi-frame series.
        """
        super()._new_video(path)
        self.frames = self.frames - self.n_frames + 1


colors = plotting.Colors() # create instance for 'from utils.plots import colors'


class MultiFrameResults:
    """
    A class for storing inference results of Multi-Frame model.
    """

    def __init__(self, results: list, n_frames: int):
        self.results = results
        self.speed = {"preprocess": None, "inference": None, "postprocess": None}
        self.save_dir = None
        self.n_frames = n_frames if n_frames else len(results)

    def __getitem__(self, idx):
        return self.results[idx]

    def plot(
        self,
        conf=True,
        line_width=None,
        im_gpu=None,
        labels=True,
        boxes=True,
    ):
        """
        Plots results for Multi-Frame model. Results are plotted on each individual frame of the multi-frame series.
        """
        plotted_images = [
            self.plot_single_frame(
                result=result,
                line_width=line_width,
                boxes=boxes,
                conf=conf,
                labels=labels,
                im_gpu=im_gpu,
            )
            for result in self.results
        ]
        return plotted_images

    # def verbose(self):
    #     super().verbose()

    def plot_single_frame(
        self,
        result: Results,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
        color_mode="class",
    ):
        """
        For multiframe task, plots detection results on a single input RGB frame.

        Changes: replaces `Annotator` with `MultiFrameAnnotator` for plotting ball traces (non-human skeleton).

        Args:
            conf (bool): Whether to plot detection confidence scores.
            line_width (float | None): Line width of bounding boxes. If None, scaled to image size.
            font_size (float | None): Font size for text. If None, scaled to image size.
            font (str): Font to use for text.
            pil (bool): Whether to return the image as a PIL Image.
            img (np.ndarray | None): Image to plot on. If None, uses original image.
            im_gpu (torch.Tensor | None): Normalized image on GPU for faster mask plotting.
            kpt_radius (int): Radius of drawn keypoints.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot labels of bounding boxes.
            boxes (bool): Whether to plot bounding boxes.
            masks (bool): Whether to plot masks.
            probs (bool): Whether to plot classification probabilities.
            show (bool): Whether to display the annotated image.
            save (bool): Whether to save the annotated image.
            filename (str | None): Filename to save image if save is True.
            color_mode (bool): Specify the color mode, e.g., 'instance' or 'class'. Default to 'class'.

        Returns:
            (np.ndarray): Annotated image as a numpy array.

        Examples:
            >>> results = model("image.jpg")
            >>> for result in results:
            ...     im = result.plot()
            ...     im.show()
        """
        assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if img is None and isinstance(result.orig_img, torch.Tensor):
            img = (result.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = result.names
        is_obb = result.obb is not None
        pred_boxes, show_boxes = result.obb if is_obb else result.boxes, boxes
        pred_masks, show_masks = result.masks, masks
        pred_probs, show_probs = result.probs, probs
        annotator = MultiFrameAnnotator(
            im=deepcopy(result.orig_img if img is None else img),
            n_frames=self.n_frames,
            line_width=line_width,
            font_size=font_size,
            font=font,
            pil=pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names,
        )

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = (
                pred_boxes.id
                if pred_boxes.id is not None and color_mode == "instance"
                else pred_boxes.cls
                if pred_boxes and color_mode == "class"
                else reversed(range(len(pred_masks)))
            )
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # Plot Detect results
        if pred_boxes is not None and show_boxes:
            for i, d in enumerate(reversed(pred_boxes)):
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {conf:.2f}" if conf else name) if labels else None
                box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(
                    box,
                    label,
                    color=colors(
                        c
                        if color_mode == "class"
                        else id
                        if id is not None
                        else i
                        if color_mode == "instance"
                        else None,
                        True,
                    ),
                    rotated=is_obb,
                )

        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(result.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        # Plot Pose results
        if result.keypoints is not None:
            for i, k in enumerate(reversed(result.keypoints.data)):
                annotator.kpts(
                    k,
                    result.orig_shape,
                    radius=kpt_radius,
                    kpt_line=kpt_line,
                    kpt_color=colors(i, True) if color_mode == "instance" else None,
                )

        # Show results
        if show:
            annotator.show(result.path)

        # Save results
        if save:
            annotator.save(filename)

        return annotator.result()

