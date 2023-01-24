import torch


class BaseMethod:
    def __init__(self,
                 model,
                 optimizer,
                 device,
                 **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def train(self, images, labels):
        output_dict = self.model(images)
        logits = output_dict['logits']
        loss = self.loss_fn(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return logits, loss



