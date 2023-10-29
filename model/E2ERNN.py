import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
import datetime

from lib.config import CONF


class E2ERNN(pl.LightningModule):
    def __init__(self, input_size=CONF.gru_model.input_size, hidden_size=CONF.gru_model.hidden_size,
                 num_layers=CONF.gru_model.num_layers):
        super().__init__()
        self.sequence_length = CONF.data.sequence_length

        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.fc = nn.Identity()

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def training_step(self, batch):
        stacked_images, label = batch
        unstacked_images = [stacked_images[:, i, :, :, :] for i in range(self.sequence_length)]

        logits = []
        for idx in range(self.sequence_length):
            logits.append(self.resnet50(unstacked_images[idx]))
        images_feature = torch.stack(logits, dim=1)

        _, hn = self.gru(images_feature)
        out = self.fc(hn[0])
        out = out.squeeze(-1)

        loss = nn.MSELoss()(out.float(), label.float())  # use L2 loss
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        stacked_images, label = batch
        unstacked_images = [stacked_images[:, i, :, :, :] for i in range(self.sequence_length)]

        logits = []
        for idx in range(self.sequence_length):
            logits.append(self.resnet50(unstacked_images[idx]))
        images_feature = torch.stack(logits, dim=1)

        _, hn = self.gru(images_feature)
        out = self.fc(hn[0])
        out = out.squeeze(-1)

        loss = nn.MSELoss()(out.float(), label.float())  # use L2 loss
        self.log('val_loss', loss, on_step=True, prog_bar=True, logger=True)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # scheduler = StepLR(optimizer, step_size=1, gamma=0.9)  # 每个epoch后，学习率乘0.9
        return [optimizer]
