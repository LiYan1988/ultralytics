from copy import deepcopy, copy

from overrides import overrides

from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils.plotting import plot_images

from .dataset import MultiFrameDataset
from .val import MultiFrameValidator
from .utils import plot_samples_multiframe


class MultiFrameTrainer(PoseTrainer):
    """
    A class extending the PoseTrainer class for training
    pose detection in video using multiple consecutive frames.
    """

    def __init__(self, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        # self.args.n_frames comes from /ultralytics/ultralytics/cfg/default.yaml
        self.n_frames = self.args.n_frames
        self.ch = self.n_frames * 3

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = PoseModel(cfg, ch=self.ch, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose)
        if weights:
            model.load(weights)

        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return MultiFrameDataset(
            n_frames=self.n_frames,
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or False,  # rectangular batches
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def plot_training_samples(self, batch, ni):
        """Plot a batch of training samples with annotated class labels, bounding boxes, and keypoints."""
        plot_samples_multiframe(batch, ni, self.save_dir, self.on_plot)

    def plot_training_labels(self):
        super().plot_training_labels()

    def get_validator(self):
        """Return an instance of the MultiFrameValidator for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return MultiFrameValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
