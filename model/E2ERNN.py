import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
import datetime

from lib.config import CONF


class E2ERNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.sequence_length = CONF.data.sequence_length
        self.model = models.resnet50(pretrained=False)
        #self.model.fc = nn.Identity()

    def test(self, batch):
        stacked_images, label = batch
        print(stacked_images.size())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        stacked_images, label = batch
        unstacked_images = [stacked_images[:, i, :, :, :] for i in range(self.sequence_length)]

        logits = []
        for idx in range(self.sequence_length):
            output = self(unstacked_images[idx])
            print(output.size())
            logits.append(output)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # scheduler = StepLR(optimizer, step_size=1, gamma=0.9)  # 每个epoch后，学习率乘0.9
        return [optimizer]