import cv2 as cv
import torch

from torch.utils.data import Dataset
from lib.config import CONF
from lib.utils import jpg_to_tensor, data_pre_processing


class SimulatorDataset(Dataset):
    def __init__(self, driving_log=CONF.PATH.SIMULATOR_STEERING_ANGLE, sequence_length=CONF.data.sequence_length,
                 transform=jpg_to_tensor):
        self.images, self.steering_angles, self.num_data_frames = data_pre_processing(driving_log)
        self.sequence_length = sequence_length
        self.transform = transform

        self.range1 = range(0, self.sequence_length)
        self.range2 = range(self.num_data_frames, self.num_data_frames + self.sequence_length)
        self.range3 = range(self.num_data_frames * 2, self.num_data_frames * 2 + self.sequence_length)

    def __len__(self):
        if len(self.images) == len(self.steering_angles):
            return len(self.images)
        else:
            print("Dataset error")

    def __getitem__(self, item):
        stacked_images = torch.zeros(self.sequence_length, 3, 160, 320)
        index = self.sequence_length

        if item in self.range1:
            while item >= 0:
                image_path = self.images[item]
                image = self.transform(cv.imread(image_path))
                stacked_images[index - 1] = image.clone()

                item = item - 1
                index = index - 1

        elif item in self.range2:
            item = item - self.num_data_frames

            while item >= 0:
                image_path = self.images[item]
                image = self.transform(cv.imread(image_path))
                stacked_images[index - 1] = image.clone()

                item = item - 1
                index = index - 1

        elif item in self.range3:
            item = item - self.num_data_frames * 2

            while item >= 0:
                image_path = self.images[item]
                image = self.transform(cv.imread(image_path))
                stacked_images[index - 1] = image.clone()

                item = item - 1
                index = index - 1

        else:
            while index >= 0:
                image_path = self.images[item]
                image = self.transform(cv.imread(image_path))
                stacked_images[index - 1] = image.clone()

                item = item - 1
                index = index - 1

        steering_angle = self.steering_angles[item]

        return stacked_images, steering_angle


def image_show(input):
    image, label = input
    print(label)
    print(" ")
    print(image.size())
    print(" ")
    print(image)

# mydataset = SimulatorDataset()
# image_show(mydataset[0])
