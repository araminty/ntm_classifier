import gc
from collections import OrderedDict
import torch
from torch import optim
from torch import nn
from ntm_classifier.check_tqdm import tqdm_check
from ntm_classifier.load_resources import (
    load_base_model, save_model_torch_script)
# from pkg_resources import resource_filename

if tqdm_check():
    from tqdm import tqdm

EPOCHS = 5


class Trainer:

    def __init__(
            self,
            dataset,
            # n_labels=20,
            batch_size=5,
            use_tqdm=None,
            epochs=EPOCHS,
            device='cpu'):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.epochs = epochs
        self.n_labels = int(dataset.na_value+1)
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size)
        self.criterion = nn.CrossEntropyLoss()
        self.use_tqdm = use_tqdm or tqdm_check()
        self.create_network()

    def create_network(self):

        self.network = nn.Sequential(OrderedDict({
            'resnet': load_base_model(),
            'fc': nn.Linear(1000, self.n_labels),
            'output': nn.LogSoftmax(dim=1)
        })).to(self.device)

        self.optimizer = optim.SGD(
            self.network.parameters(), lr=0.005, momentum=0.9)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=0.01,
            steps_per_epoch=len(self.dataloader),
            epochs=self.epochs)

    def train_epoch(self):
        # torch.cuda.empty_cache()
        running_loss = 0.0

        if self.use_tqdm:
            loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
            loop.desc = ('train batch: 0| Cross Entropy: 0.0')
        else:
            loop = enumerate(self.dataloader)

        for i, data in loop:
            inputs = data['image'].squeeze().to(self.device)
            targets = data['result'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.network(inputs.to(self.device))
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            running_loss += loss.item()
            if self.use_tqdm:
                loop.desc = ('train batch: %2d| Cross Entropy: %.3f' %
                             (i, running_loss / (i + 1)))

        return running_loss / min(1, len(self.dataloader))

    def train(self):
        self.optimizer = optim.SGD(
            self.network.parameters(), lr=0.005, momentum=0.9)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=0.01,
            steps_per_epoch=len(self.dataloader),
            epochs=self.epochs)

        for i in range(self.epochs):
            training_loss = str(self.train_epoch())
            print(f"Epoch {i}| Training Loss: {training_loss}")
            gc.collect()
            torch.cuda.empty_cache()

    def save(self, name=None):
        if name is None:
            name = f"n_{self.n_labels}_epochs_{self.epochs}_model.pt"
        # filepath = resource_filename('ntm_data', name)
        save_model_torch_script(self.network, name=name)
