import pytorch_lightning as pl
from lib.datamodule import SimulatorDataModule
from model.E2ERNN import E2ERNN

# prepare dataset and dataloader
data = SimulatorDataModule()

model = E2ERNN()

# start training
trainer = pl.Trainer(accelerator='gpu',
                     devices=1,
                     max_epochs=10,
                     log_every_n_steps=5,
                     #limit_train_batches=15
                     )

trainer.fit(model, data)

# tensorboard --logdir=/home/jiachen/SelfDrivingCars_GRU/scripts/lightning_logs