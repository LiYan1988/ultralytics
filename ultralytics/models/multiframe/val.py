import torch

from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.utils.plotting import plot_images, output_to_target

from .utils import plot_samples_multiframe, plot_images_multiframe


class MultiFrameValidator(PoseValidator):
    """
    A class extending the PoseValidator class for validation based on multi-frame pose model.
    """

    def plot_val_samples(self, batch, ni):
        plot_samples_multiframe(batch, ni, self.save_dir, self.on_plot)

    def plot_predictions(self, batch, preds, ni):
        """
        Plot predictions on every frame of the multi-frame series.
        """
        pred_kpts = torch.cat([p[:, 6:].view(-1, *self.kpt_shape) for p in preds], 0)
        n_frames = batch['img'].shape[1] // 3
        for i in range(n_frames):
            plot_images_multiframe(
                batch["img"][:, i * 3:i * 3 + 3, ...],
                n_frames,
                *output_to_target(preds, max_det=self.args.max_det),
                kpts=pred_kpts,
                paths=batch["im_file"],
                fname=self.save_dir / f"val_batch{ni}_frame{i}_pred.jpg",
                names=self.names,
                on_plot=self.on_plot,
                threaded=False, # Disable threading in development
            )  # pred
