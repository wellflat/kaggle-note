from typing import Any, Dict, List, Tuple, Union
import torch
from torch import nn, optim, Tensor
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from lenet import LeNet
from config import TrainingConfig

class Classifier(pl.LightningModule):
    net: nn.Module
    criterion: nn.CrossEntropyLoss
    config: TrainingConfig

    def __init__(self, config:TrainingConfig):
        super().__init__()
        self.config = config
        self.net = LeNet(num_classes=10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x:Tensor) -> Tensor:
        return self.net.forward(x)

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.StepLR]]:
        optimizer = optim.Adam(self.parameters(), lr=self.config.lr)
        scheduler = StepLR(optimizer, step_size=self.config.step_size, gamma=self.config.gamma)
        return [optimizer], [scheduler]
    
    def training_step(self, batch:Tuple[Tensor, Tensor], batch_idx:int) -> Tensor:
        inputs, targets = batch
        logits: Tensor = self.net(inputs)
        loss: Tensor = self.criterion(logits, targets)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs: Tensor) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('loss', avg_loss, prog_bar=True)
    
    def validation_step(self, batch:Tuple[Tensor, Tensor] , batch_idx:int) -> Dict[str, Any]:
        inputs, targets = batch
        logits: Tensor = self.net(inputs)
        preds = logits.argmax(1)
        accuracy = sum(preds == targets) / len(targets)
        return {'val_acc': accuracy}
    
    def validation_epoch_end(self, outputs: Tensor) -> None:
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('val_acc', avg_acc, prog_bar=True)

    def test_step(self, *args, **kwargs):
        pass

    def test_epoch_end(self, outputs) -> None:
        pass


    