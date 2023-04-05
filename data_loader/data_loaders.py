import os
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader


# TODO: refactor. No need to have two datasets here
class DataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        augment: bool = True,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        self.data_dir = data_dir
        self.dataset = HasselbladDataset(self.data_dir, augment)
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


class HasselbladDataset(Dataset):
    def __init__(self, data_dir, augment: bool = True):
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
        self.augment = augment

        self.to_tensor = transforms.ToTensor()
        self.input_dir = os.path.join(data_dir, "input")
        self.label_dir = os.path.join(data_dir, "label")
        self.images = sorted(
            os.listdir(self.input_dir), key=lambda x: int(x.split(".")[0])
        )
        self.target = sorted(
            os.listdir(self.label_dir), key=lambda x: int(x.split(".")[0])
        )

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.input_dir, self.images[index]))
        image = self.to_tensor(image)
        target = Image.open(os.path.join(self.label_dir, self.target[index]))
        target = self.to_tensor(target)
        data = torch.stack([image, target], dim=0)
        if self.augment:
            data = self.transform(data)
        return data[0], data[1]

    def __len__(self):
        return len(self.images)
