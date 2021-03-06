
from typing import Any, Dict, List, Tuple
import os
import torch
from torch import nn, optim
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.transforms import Compose
from PIL.Image import Image as PILImage
from tqdm.autonotebook import tqdm
from lenet import LeNet


class Classifier:
    net: LeNet
    criterion: nn.CrossEntropyLoss
    optimizer: optim.SGD
    scheduler: optim.lr_scheduler.StepLR
    transformer: Compose
    logger: SummaryWriter

    def __init__(self, conf: Dict[str, Any]):
        super().__init__()
        self.net = LeNet(conf.get('num_classes', 10))
        self.criterion = nn.CrossEntropyLoss()
        self.__build_transformer()
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=conf.get('base_lr', 0.01),
            momentum=conf.get('momentum', 0.9),
            weight_decay=5e-4
        )
        self.schedular = StepLR(self.optimizer, step_size=25, gamma=0.1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False
        self.net.to(self.device)
        self.best_acc = 0.0
        self.start_epoch = 0
        self.logger = SummaryWriter()
        
    
    def fit(self, loaders: Dict[str, DataLoader], epochs: int, resume: bool=False) -> None:
        best_acc = 0.0
        start_epoch = 0
        if resume:
            start_epoch, best_acc = self.load_checkpoint()
        
        progress = tqdm(
            range(start_epoch, start_epoch + epochs),
            total=epochs, initial=start_epoch
        )
        print(f'start training on {self.device.type}')
        progress.set_description('Epoch')
        for epoch in progress:
            loss = self.__train(loaders['train'])
            self.logger.add_scalar('training loss', loss, epoch)
            val_acc = self.__validate(loaders['val'])
            self.logger.add_scalar('validation accuracy', val_acc, epoch)
            self.schedular.step()
            lr = self.schedular.get_last_lr()[0]
            self.logger.add_scalar('learning rate', lr, epoch)
            
            if val_acc > best_acc:
                tqdm.write('saving checkpoint...')
                best_acc = val_acc
                state = {
                    'net': self.net.state_dict(),
                    'acc': best_acc,
                    'epoch': epoch
                }
                self.save_checkpoint(state)
            
            progress.set_postfix({'loss': loss, 'acc': val_acc, 'lr': lr})


    def __train(self, loader: DataLoader) -> float:
        self.net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(enumerate(loader), total=len(loader), leave=False)
        progress.set_description('Train')
        for batch_idx, (inputs, targets) in progress:
            inputs = inputs.reshape(inputs.shape[0], 1, 28, 28)
            # normalize
            inputs = (inputs/255 - 0.5) / 0.5
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100.*correct/total

            progress.set_postfix({
                'loss': (running_loss/(batch_idx+1)),
                'acc': accuracy
            })

        epoch_loss = running_loss / len(loader)
        return epoch_loss
    

    def __validate(self, loader: DataLoader) -> float:
        self.net.eval()
        correct = 0
        total = 0
        progress = tqdm(enumerate(loader), total=len(loader), leave=False)
        progress.set_description('Val  ')
        for batch_idx, (inputs, targets) in progress:
            inputs = inputs.reshape(inputs.shape[0], 1, 28, 28)
            inputs = (inputs/255 - 0.5) / 0.5
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.inference_mode():
                outputs: torch.Tensor = self.net(inputs)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100.*correct/total

            progress.set_postfix({'acc': accuracy})
                
        epoch_accuracy = 100.*correct / len(loader.dataset)
        return epoch_accuracy


    def test(self, loader: DataLoader) -> None:
        self.net.eval()
        device = self.device
        correct = 0
        total = 0

        progress = tqdm(enumerate(loader), total=len(loader))
        progress.set_description('Test')
        for batch_idx, (inputs, targets) in progress:
            inputs = inputs.reshape(inputs.shape[0], 1, 28, 28)
            inputs = (inputs/255 - 0.5) / 0.5
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.inference_mode():
                outputs: torch.Tensor = self.net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100.*correct/total

            progress.set_postfix({'acc': accuracy})
        
        test_accuracy = 100.*correct / len(loader.dataset)
        print(f'test accuracy: {test_accuracy}')

    
    def test_nolabel(self, loader: DataLoader) -> torch.Tensor:
        self.net.eval()
        device = self.device

        result = []
        progress = tqdm(enumerate(loader), total=len(loader))
        progress.set_description('Test')
        for batch_idx, (inputs,) in progress:
            inputs = inputs.reshape(inputs.shape[0], 1, 28, 28)
            inputs = (inputs/255 - 0.5) / 0.5
            inputs = inputs.to(device)
            with torch.inference_mode():
                outputs: torch.Tensor = self.net(inputs)
            
            _, predicted = outputs.max(1)
            result.append(predicted)
        
        return torch.cat(result)


    def predict(self, x: PILImage) -> torch.Tensor:
        #self.net.eval()
        #x.to(self.device)
        #return self.net(x)
        x = self.transformer(x)
        x = x.unsqueeze(0)  # type: ignore
        with torch.inference_mode():
            outputs = self.net(x)

        _, preds = torch.max(outputs, 1)  # type: ignore
        label = int(preds)
        output = softmax(input=outputs, dim=1)[:, 1]
        score = float(output.cpu()) if label == 1 else 1 - float(output.cpu())
        confidence = float('{:.3f}'.format(score))
        return (label, confidence)


    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))


    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)


    def load_checkpoint(self) -> Tuple[int, float]:
        filepath = f'./checkpoint/{self.__class__.__name__}_model.pth'
        checkpoint = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        return (int(checkpoint['epoch']), checkpoint['acc'])


    def save_checkpoint(self, state: Dict[str, Any]) -> None:
        os.makedirs('./checkpoint', exist_ok=True)
        filepath = f'./checkpoint/{self.__class__.__name__}_model.pth'
        torch.save(state, filepath)


    def __build_transformer(self) -> None:
        #image_size = self.config['image_size']
        image_size = 28
        self.transformer = transforms.Compose([
            #transforms.Resize((image_size, image_size)),
            #transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])



