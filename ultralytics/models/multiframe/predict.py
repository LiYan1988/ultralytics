from pathlib import Path
import re

import numpy as np
import torch
import cv2

from ultralytics.utils.checks import check_imgsz
from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.engine.predictor import STREAM_WARNING
from ultralytics.utils import LOGGER, ops, DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr
from ultralytics.engine.results import Results
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes

from .dataset import MultiFrameVideoLoader, MultiFrameResults
from .augment import MultiFrameLetterBox


class MultiFramePredictor(PosePredictor):

    def __init__(self, **kwargs):
        self.n_frames = kwargs['overrides'].get('n_frames', 1)
        super().__init__(**kwargs)

    def setup_source(self, source):
        """
        Sets up source and inference mode for MultiFrame model with video input.
        """
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = None # This is used only for classify task
        self.dataset = MultiFrameVideoLoader(source, n_frames=self.n_frames, batch=self.args.batch)
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    def pre_transform(self, im):
        """
        Pre-transform input images before inference of the multi-frame model.

        Args:
            im (list[list[np.ndarray]]): [[(h, w, 3) x n_frames] x batch_size]

        Returns:
            (list): A list of transformed images.
        """
        same_shapes = len({x.shape for b in im for x in b}) == 1
        letterbox = MultiFrameLetterBox(self.imgsz, auto=same_shapes and self.model.pt, stride=self.model.stride)
        # BRG to RGB
        # TODO: double check this is correct.
        return [letterbox(images=im_)[...,::-1] for im_ in im]
        # return [letterbox(images=im_) for im_ in im]

    def preprocess(self, im: list[np.ndarray]):
        """
        Prepares input image before inference for the multi-frame model.

        Args:
            im List(np.ndarray): [(HWC) x B] for list.
        """
        im = np.stack(self.pre_transform(im))
        # BHWC to BCHW, (n, 3, h, w). BRG to RGB is performed in self.pre_transform
        im = im.transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def postprocess(self, preds, img, orig_imgs):
        """
        For Multi-Frame model, return detection results for a given input image or list of images.
        """
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            orig_img_shape = orig_img[0].shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img_shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img_shape)
            results.append(
                MultiFrameResults(
                    results=[
                        Results(
                            img_,
                            path=img_path,
                            names=self.model.names,
                            boxes=pred[:, :6],
                            keypoints=pred_kpts
                        )
                        for img_ in orig_img
                    ]
                )
            )
        return results

    def write_results(self, i, p, im, s):
        """For Multi-Frame model, write detection results to file or directory."""
        string = ""  # print string
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 0 if frame undetermined

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # used in other locations
        # TODO: Appending detection results from the multiframe model?
        #   Skip appending detection results to the output string for now.
        # string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # Add predictions to image
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else im[i],
            )

        # Save results, discard other savings in the original method, only save videos.
        if self.args.save:
            self.save_predicted_images(str(self.save_dir / p.name), frame)

        return string

    def save_predicted_images(self, save_path='', frame=0):
        """For Multi-Frame model, save video predictions as mp4 at specified path."""
        # TODO: Plot all frames in a video
        # Only plot the last image for now
        im = self.plotted_img[-1]

        # Save videos and streams
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = f'{save_path.split(".", 1)[0]}_frames/'
            if save_path not in self.vid_writer:  # new video
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # integer required, floats produce error in MP4 codec
                    frameSize=(im.shape[1], im.shape[0]),  # (width, height)
                )

            # Save video
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{frame}.jpg", im)

        # Save images
        else:
            cv2.imwrite(save_path, im)
