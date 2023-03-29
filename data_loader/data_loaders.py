import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader


class DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([800,800])
        ])
        self.data_dir = data_dir
        self.dataset = HasselbladDataset(self.data_dir, trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class HasselbladDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.input_dir = os.path.join(data_dir, "input")
        self.label_dir = os.path.join(data_dir, "label")
        self.images = sorted(os.listdir(self.input_dir))
        self.target = sorted(os.listdir(self.label_dir))

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.input_dir, self.images[index]))
        target = Image.open(os.path.join(self.label_dir, self.target[index]))
        image = self.transform(image)
        target = self.transform(target)

        return image, target

    def __len__(self):
        return len(self.images)
