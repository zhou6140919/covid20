import os

from PIL import Image
from torch.utils import data


class Xray(data.Dataset):
    def __init__(self, root, transforms):
        self.transforms = transforms
        p_root = root + '/positive/'
        n_root = root + '/negative/'
        imgs_p = [os.path.join(p_root, img) for img in os.listdir(p_root)]
        imgs_n = [os.path.join(n_root, img) for img in os.listdir(n_root)]
        imgs = imgs_p + img_n
        self.imgs = imgs

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if 'positive' in img_path.split('/') else 0
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
