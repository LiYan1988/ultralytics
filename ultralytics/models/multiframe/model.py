from ultralytics.engine.model import Model

from .train import MultiFrameTrainer


class MultiFrameModel(Model):

    def __init__(self, model='yolov8n-pose.pt'):
        super().__init__(model=model, task='pose', verbose=True)

    @property
    def task_map(self) -> dict:
        return {
            "pose": {
                "trainer": MultiFrameTrainer,
            }
        }
