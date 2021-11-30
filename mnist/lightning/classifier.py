from typing import Tuple
from torch import nn, optim, Tensor
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl


class Classifier(pl.LightningModule):
    net: nn.Module

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.net.forward(x)
    
    def configure_optimizers(self) -> Tuple[optim.Adam, StepLR]:
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
        return (optimizer, scheduler)
    
    def training_step(self, batch, batch_idx) -> Tensor:
        self.log("training_step", batch_idx)
        inputs, targets = batch
        logits: Tensor = self.net(inputs)
        loss = self.criterion(logits, targets)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits: Tensor = self.net(inputs)
        _, preds = logits.max(1)
        loss = self.criterion(logits, targets)
        self.log('val_loss', loss)

    def training_epoch_end(self, loss: Tensor) -> None:
        pass

    def validation_epoch_end(self, loss: Tensor) -> None:
        pass


    