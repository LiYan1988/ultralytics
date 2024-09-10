from ultralytics.engine.model import Model
from ultralytics.nn.tasks import PoseModel
from ultralytics.models.yolo.pose.predict import PosePredictor

from .train import MultiFrameTrainer
from .val import MultiFrameValidator
from .predict import MultiFramePredictor


class MultiFrame(Model):

    def __init__(self, n_frames, model='yolov8n-pose.pt'):
        super().__init__(model=model, task='pose', verbose=True)
        self.n_frames = n_frames # self.n_frames is not being used during training?

    @property
    def task_map(self) -> dict:
        return {
            "pose": {
                "model": PoseModel,
                "trainer": MultiFrameTrainer,
                "validator": MultiFrameValidator,
                "predictor": MultiFramePredictor,
            }
        }
