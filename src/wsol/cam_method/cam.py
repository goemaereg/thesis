import numpy as np
from .base_cam import BaseCAM


class CAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            CAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        if self.model.__class__.__name__ != 'VggCam':
            raise NotImplementedError
        return self.model.fc.weight[targets]
