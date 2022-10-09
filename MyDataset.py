import os
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# 构建数据集
train_transform = transforms.Compose([transforms.Resize((150, 450)), transforms.ToTensor()])


def transform_label(fn):
    cl = fn.split('_')[-1][:-4]
    datalist = ''.join(cl.split('-'))
    length = len(datalist)
    datalist = list((map(int, datalist))) + [0] * (150 - length)
    return np.array(datalist), length


class MyDataset(Dataset):
    def __init__(self, data_path, re_str='.*', sample=1.0, transform=train_transform):
        rr = re.compile(re_str)
        self.source_path = data_path
        assert os.path.exists(self.source_path), 'the path :%s not exist' % self.source_path
        self.transform = transform
        image_names = []
        for f in os.listdir(self.source_path):
            if rr.search(f):
                image_names.append(f)
        if 0 < sample < 1.0:  # 1/10
            _size = int(len(image_names) * sample)
            self.image_names = image_names[:_size]
        else:
            self.image_names = list(image_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        fn = self.image_names[index]
        image = self.transform(Image.open(self.source_path + '/' + fn))
        label, length = transform_label(fn)
        return image, label, length
