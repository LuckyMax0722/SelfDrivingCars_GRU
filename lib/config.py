import os
from easydict import EasyDict

CONF = EasyDict()

# Main Path
CONF.PATH = EasyDict()
CONF.PATH.BASE = '/home/jiachen/SelfDrivingCars_GRU'  # TODO: change this
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, 'data')
CONF.PATH.OUTPUT_MODEL = os.path.join(CONF.PATH.BASE, 'output/model/')

# Simulator Data
CONF.PATH.SIMULATOR_STEERING_ANGLE = os.path.join(CONF.PATH.DATA, "driving_log.csv")

# Dataset
CONF.data = EasyDict()
CONF.data.source = 'Download'  # 'Download', 'Simulator'  # TODO: change this
CONF.data.sequence_length = 5

# Datamodule
CONF.datamodule = EasyDict()
CONF.datamodule.batch_size = 8
CONF.datamodule.train_val_split = 0.9

# GRU Model
CONF.gru_model = EasyDict()
CONF.gru_model.input_size = 2048
CONF.gru_model.hidden_size = 512
CONF.gru_model.num_layers = 1



# Best Model
CONF.model = EasyDict()
CONF.model.best_model = '/home/jiachen/SelfDrivingCars/output/model/model_Resnet50.pth'  # TODO: change this
