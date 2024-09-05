from ultralytics.engine.model import Model
from ultralytics.nn.tasks import PoseModel

from .train import MultiFrameTrainer
from .val import MultiFrameValidator


class MultiFrame(Model):

    def __init__(self, model='yolov8n-pose.pt'):
        super().__init__(model=model, task='pose', verbose=True)

    @property
    def task_map(self) -> dict:
        return {
            "pose": {
                "model": PoseModel,
                "trainer": MultiFrameTrainer,
                "validator": MultiFrameValidator,
            }
        }
