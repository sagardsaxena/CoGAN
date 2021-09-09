import cv2
import os
import numpy as np
import torch.utils.data as data
import torch


class IMAGEDIR(data.Dataset):

    def __init__(self, root, num_samples, label, train=True, transform=None, seed=None,
                 scale=2, bias=-1):
        self.train = train
        self.root = root
        self.num_samples = num_samples
        self.transform = transform
        self.label = torch.LongTensor([np.int64(label)])
        self.data = np.array(os.listdir(root))
        self.scale = scale
        self.bias = bias

        if seed is not None: np.random.seed(seed)

        total_num_samples = len(self.data)
        if self.num_samples > total_num_samples: self.num_samples = total_num_samples

        indices = np.arange(total_num_samples)
        np.random.shuffle(indices)
        self.data = self.data[indices[:self.num_samples]]

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.root, self.data[index]))
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None: img = self.transform(img)
        img = self.scale*img + self.bias
        return img, self.label

    def __len__(self):
        return self.num_samples

