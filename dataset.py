from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection, VOCSegmentation
from torchvision.transforms import Resize, ToTensor, Compose
import os
import pickle
import numpy as np
import cv2
from PIL import Image


class AnimalDataset(Dataset):
    def __init__(self, root, is_train, transform=None):
        if is_train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                           "squirrel"]
        self.image_paths = []
        self.labels = []
        for index, category in enumerate(self.categories):
            subdir_path = os.path.join(data_path, category)
            for file_name in os.listdir(subdir_path):
                self.image_paths.append(os.path.join(subdir_path, file_name))
                self.labels.append(index)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        # image = cv2.imread(self.image_paths[item])
        image = Image.open(self.image_paths[item]).convert("RGB")
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        return image, label


class MyCifar10(Dataset):
    def __init__(self, root, is_train):
        if is_train:
            data_files = [os.path.join(root, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(root, "test_batch")]
        self.all_images = []
        self.all_labels = []
        for data_file in data_files:
            with open(data_file, "rb") as fo:
                data = pickle.load(fo, encoding="bytes")
                images = data[b'data']
                labels = data[b'labels']
                self.all_images.extend(images)
                self.all_labels.extend(labels)

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, index):
        image = self.all_images[index]
        image = np.reshape(image, (3, 32, 32)).astype(np.float32)/255.
        label = self.all_labels[index]
        return image, label

class VOCDatasetDetection(VOCDetection):
    def __init__(self, root, year, image_set, download, transform):
        super().__init__(root, year, image_set, download, transform)
        self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']

    def __getitem__(self, item):
        image, data = super().__getitem__(item)
        all_bboxes = []
        all_labels = []
        for obj in data["annotation"]["object"]:
            xmin = int(obj["bndbox"]["xmin"])
            ymin = int(obj["bndbox"]["ymin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymax = int(obj["bndbox"]["ymax"])
            all_bboxes.append([xmin, ymin, xmax, ymax])
            all_labels.append(self.categories.index(obj["name"]))
        all_bboxes = torch.FloatTensor(all_bboxes)
        all_labels = torch.LongTensor(all_labels)
        target = {
            "boxes": all_bboxes,
            "labels": all_labels
        }
        return image, target

class VOCDatasetSegmentation(VOCSegmentation):
    def __init__(self, root, year, image_set, download, transform=None, target_transform=None):
        super().__init__(root, year, image_set, download, transform, target_transform)
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']


    def __getitem__(self, item):
        image, target = super().__getitem__(item)
        target = np.array(target, np.int64)
        target[target == 255] = 0
        return image, target
if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    train_dataset = AnimalDataset(root="data/animals", is_train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        num_workers=8,
        shuffle=True,
        drop_last=True
    )
    for epoch in range(100):
        for images, labels in train_dataloader:
            print(images.shape, labels)
