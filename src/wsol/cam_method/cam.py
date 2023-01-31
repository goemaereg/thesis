import numpy as np
from .base_cam import BaseCAM


class CAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(CAM,self).__init__(
            model,
            target_layers,
            use_cuda=use_cuda,
            reshape_transform=reshape_transform,
            uses_gradients=False)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        if self.model.__class__.__name__ != 'VggCam':
            raise NotImplementedError
        _targets = np.asarray(list(map(lambda x: x.category, targets)))
        _weights = self.model.fc.weight.numpy(force=True)
        weights = _weights[_targets]
        return weights