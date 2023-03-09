import torch
import torch.nn as nn
from .base_cam import BaseCAM


class ScoreCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            device='cuda',
            reshape_transform=None):
        super(ScoreCAM, self).__init__(model,
                                       target_layers,
                                       device=device,
                                       reshape_transform=reshape_transform,
                                       uses_gradients=False)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        # input_tensor: shape (batch, channels, H, W)
        with torch.no_grad():
            # TODO: test with do_upsample.to(device) !!
            # TODO: test with Batch = 1
            # do_upsample = torch.nn.UpsamplingBilinear2d(
            #     size=input_tensor.shape[-2:]) # Todo: deprecated -> use interpolate
            # activation_tensor : shape(image batch size, activation channels, H, W)
            activation_tensor = torch.from_numpy(activations).to(self.device)
            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 64 #16
            print(f'scorecam: BATCHSIZE = {BATCH_SIZE}')
            scores = []
            it_inputs = zip(targets, input_tensor, activation_tensor)# upsampled)
            for in_index, (target, in_tensor, act_tensor) in enumerate(it_inputs):
                # compute upsampled activations here to reduce memory requirements by factor of input batch size
                # act_tensor : shape (activation channels, H, W)
                act_upsampled = nn.functional.interpolate(
                    act_tensor[None, :, :, :],
                    size=input_tensor.shape[-2:],
                    mode='bilinear', align_corners=True).squeeze(dim=0)
                maxs = act_upsampled.view(act_upsampled.size(0), -1).max(dim=-1)[0]
                mins = act_upsampled.view(act_upsampled.size(0), -1).min(dim=-1)[0]
                # maxs, mins: shape(activation channels, 1, 1)
                maxs, mins = maxs[:, None, None], mins[:, None, None]
                # upsampled: shape(activation channels, H, W)
                act_tensor = (act_upsampled - mins) / (maxs - mins)
                # replaces nan values with 0.0 where maxs == mins (can happen for channels without activations)
                act_tensor = torch.nan_to_num(act_tensor)
                # compute input * activation here to reduce memory requirements by factor of input batch size
                # in_tensor: shape(RGB channels, H, W) -> shape(1, RGB channels, H, W)
                # act_tensor: shape(activation channels, H, W) -> shape(activation channels, 1, H, W)
                # resulting in_tensor: shape(activation channels, RGB channels, H, W)
                in_tensor = in_tensor[None, :, :, :] * act_tensor[:, None, :, :]
                it_acts_batches = range(0, in_tensor.size(0), BATCH_SIZE)
                for act_index  in it_acts_batches:
                    # act_channels_batch: shape(batch size act channels, RGB channels, H, W)
                    act_channels_batch = in_tensor[act_index: act_index + BATCH_SIZE]
                    # logits: shape(batch size act channels, num classes)
                    logits = self.model(act_channels_batch)['logits']
                    outputs = target(logits).cpu().tolist()
                    # outputs: shape(batch size act channels)
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            # scores: shape(image batch size, activation channels
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights
