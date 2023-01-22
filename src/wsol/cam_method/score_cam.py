import torch
import tqdm
from .base_cam import BaseCAM


class ScoreCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            use_cuda=False,
            reshape_transform=None):
        super(ScoreCAM, self).__init__(model,
                                       target_layers,
                                       use_cuda,
                                       reshape_transform=reshape_transform,
                                       uses_gradients=False)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        # input_tensor: shape (batch, H, W, channels)
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(
                size=input_tensor.shape[-2:])
            activation_tensor = torch.from_numpy(activations)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()
            # activation tensor: shape (batch, channels, H, W)
            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]
            # maxs, mins: shape(batch, channels, 1, 1)
            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            # upsampled: shape(batch, channels, H, W)
            upsampled = (upsampled - mins) / (maxs - mins)
            # input tensor: shape(batch, RGB channels, H, W) -> shape(batch, 1, RGB channels, H, W)
            # upsampled: shape(batch, channels, H, W) -> shape(batch, channels, 1, H, W)
            # input_tensors = input_tensor[:, None,
            #                              :, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 64 #16

            scores = []
            it_inputs = zip(targets, input_tensor, upsampled)
            for target, in_tensor, act_tensor in tqdm.tqdm(it_inputs, total=len(targets), desc='scorecam images'):
                # compute input * activation here to reduce memory requirements by factor of input batch size
                # input tensor: shape(RGB channels, H, W) -> shape(1, RGB channels, H, W)
                # upsampled: shape(activation channels, H, W) -> shape(activation channels, 1, H, W)
                # resulting in_tensor: shape(activation channels, RGB channels, H, W)
                in_tensor = in_tensor[None, :, :, :] * act_tensor[:, None, :, :]
                it_acts = range(0, in_tensor.size(0), BATCH_SIZE)
                for i in tqdm.tqdm(it_acts, total=len(it_acts), desc='scorecam activation batches'):
                    batch = in_tensor[i: i + BATCH_SIZE]
                    logits = self.model(in_tensor)['logits']
                    outputs = target(logits).cpu().tolist()
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights
