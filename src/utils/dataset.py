import os
from ast import literal_eval

import numpy as np
import torch
from PIL import Image
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision.transforms import Resize


class HeiChole(Dataset):
    def __init__(self, surgery_folder: str,
                 num_classes: int,
                 width: int, height: int,
                 transform: None or Module,
                 label_file="Ins.csv",
                 load_to_ram=False):

        super(HeiChole, self).__init__()

        self.surgery_folder = surgery_folder
        self.num_classes = num_classes
        self.tranform = transform
        self.width = width
        self.height = height
        self.load_to_ram = load_to_ram
        self.files: list = []
        self.labels: dict = {}

        with open(os.path.join(surgery_folder, label_file)) as csv:
            labels_d = {int(label[:-1].split(',')[0]): [int(b) for b in label[:-1].split(',')[1:]]
                        for label in csv.readlines()}

        # get removed white frame indices
        with open(surgery_folder + "/removed_frames.txt", "r") as file:
            removed_frames = file.readline()
            removed_frames = literal_eval(removed_frames)

        for file in sorted([f for f in os.listdir(surgery_folder) if f.lower().endswith('.png')]):
            key = int(file[:-4])
            if key in labels_d and key not in removed_frames:
                self.labels[file] = np.asarray(labels_d[key], dtype=bool)[:self.num_classes]
        self.files = sorted(self.labels.keys())

        if self.load_to_ram:
            self.ram = {}

    def _resize(self, img: np.ndarray) -> np.ndarray:
        img = img.transpose([2, 0, 1])
        c, h, w = img.shape
        self.height2 = int(self.width * (h / w))
        offset_y = (self.height - self.height2) // 2
        img_y = Resize((self.height2, self.width))(torch.tensor(img.copy())).numpy()

        if offset_y < 0:
            offset_y *= -1
            img = img_y[:, offset_y:offset_y + self.height, :]
        elif offset_y > 0:
            img = np.zeros((c, self.height, self.width), dtype=img.dtype)
            img[:, offset_y:offset_y + img_y.shape[1], :] = img_y
        else:  # offset_y == 0
            img = img_y

        return img

    def load_frame(self, path):
        if self.load_to_ram:  # if frames should be stored in the ram
            if path not in self.ram:  # check if already there
                self.ram[path] = self._resize(np.asarray(Image.open(path), dtype=np.uint8))  # if not load to ram
            return self.ram[path]  # return from ram

        return self._resize(np.asarray(Image.open(path), dtype=np.uint8))  # default load & return from file

    def __getitem__(self, i: int or str):
        file = None

        if isinstance(i, str) and i in self.files:  # frame as index
            file = i
        if isinstance(i, int):  # running number as index
            file = self.files[i]

        if file is None:
            raise IndexError(f"Index '{i}' was not found!")

        label = self.labels[file]
        frame = self.load_frame(os.path.join(self.surgery_folder, file))

        # print(label)
        # print(frame.shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(np.moveaxis(frame, 0, -1))
        # plt.show()

        if self.tranform is not None:
            frame = self.tranform(frame.transpose([1, 2, 0]))
        return file, frame, label

    def __len__(self):
        return len(self.files)


def map_cholec_labels_to_heichole(cholec_labels):
    cholec_labels = np.array([int(label) for label in cholec_labels])  # transform to int
    permute_idxs = [0, 3, 2, 4, 5, 6, 1]
    # print("chol", cholec_labels)
    heichole_labels = cholec_labels[permute_idxs]  # map label onto each other
    heichole_labels[2] = cholec_labels[1] or cholec_labels[2]  # 1 and 2 belong to the same class
    heichole_labels[6] = 0  # last label doesnt exist in Cholec80
    # print("hei", heichole_labels)

    return heichole_labels


class Cholec80(Dataset):
    def __init__(self, surgery_folder: str,
                 num_classes: int,
                 width: int, height: int,
                 transform: None or Module,
                 label_file="Ins.csv",
                 load_to_ram=False):

        super(Cholec80, self).__init__()

        self.surgery_folder = surgery_folder
        self.num_classes = num_classes
        self.tranform = transform
        self.width = width
        self.height = height
        self.load_to_ram = load_to_ram
        self.files: list = []
        self.labels: dict = {}

        with open(os.path.join(surgery_folder, label_file)) as csv:
            labels_d = {int(label[:-1].split(',')[0]):
                        [b for b in map_cholec_labels_to_heichole(label[:-1].split(',')[1:])]
                        for label in csv.readlines()}

        for file in sorted([f for f in os.listdir(surgery_folder) if f.lower().endswith('.png')]):
            key = int(file[:-4]) * 25
            if key in labels_d:
                self.labels[file] = np.asarray(labels_d[key], dtype=bool)[:self.num_classes]

        self.files = sorted(self.labels.keys())

        if self.load_to_ram:
            self.ram = {}

    def _resize(self, img: np.ndarray) -> np.ndarray:
        img = img.transpose([2, 0, 1])
        c, h, w = img.shape
        self.height2 = int(self.width * (h / w))
        offset_y = (self.height - self.height2) // 2
        img_y = Resize((self.height2, self.width))(torch.tensor(img.copy())).numpy()

        if offset_y < 0:
            offset_y *= -1
            img = img_y[:, offset_y:offset_y + self.height, :]
        elif offset_y > 0:
            img = np.zeros((c, self.height, self.width), dtype=img.dtype)
            img[:, offset_y:offset_y + img_y.shape[1], :] = img_y
        else:  # offset_y == 0
            img = img_y

        return img

    def load_frame(self, path):
        if self.load_to_ram:  # if frames should be stored in the ram
            if path not in self.ram:  # check if already there
                self.ram[path] = self._resize(np.asarray(Image.open(path), dtype=np.uint8))  # if not load to ram
            return self.ram[path]  # return from ram

        return self._resize(np.asarray(Image.open(path), dtype=np.uint8))  # default load & return from file

    def __getitem__(self, i: int or str):
        file = None

        if isinstance(i, str) and i in self.files:  # frame as index
            file = i
        if isinstance(i, int):  # running number as index
            file = self.files[i]

        if file is None:
            raise IndexError(f"Index '{i}' was not found!")

        label = self.labels[file]
        frame = self.load_frame(os.path.join(self.surgery_folder, file))

        # print(label)
        # print(frame.shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(np.moveaxis(frame, 0, -1))
        # plt.show()

        if self.tranform is not None:
            frame = self.tranform(frame.transpose([1, 2, 0]))
        return file, frame, label

    def __len__(self):
        return len(self.files)


class Cohort(Dataset):
    def __init__(self, data_root: str, client_data: list, num_classes: int, **kwargs):
        super(Cohort, self).__init__()
        self.surgery_datasets = {}

        # right now only assumes data from one dataset, else surgery names could include duplicates
        for dataset_name, surgery_names in client_data.items():
            for name in surgery_names:
                name = str(name)
                if dataset_name == "HeiChole":
                    self.surgery_datasets[dataset_name + name] = \
                        HeiChole(os.path.join(data_root, f"train/HeiChole/" + name), num_classes, **kwargs)

                elif dataset_name == "Cholec80":
                    self.surgery_datasets[dataset_name + name] = \
                        Cholec80(os.path.join(data_root, f"train/Cholec80/" + name), num_classes, **kwargs)

            self.surgeries = sorted(self.surgery_datasets.keys())
            self.files = [f"{surgery}/{file}"
                          for surgery in self.surgeries
                          for file in self.surgery_datasets[surgery].files]

    def __getitem__(self, item):
        path = self.files[item]
        surgery, file = path.split('/')
        surgery_file, frame, label = self.surgery_datasets[surgery][file]
        assert file == surgery_file
        return path, frame, label

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return f"Cohort with {str(len(self.files))} elements."
