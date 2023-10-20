import os
import random
import numpy as np
import cv2
from torch.utils.data import Dataset

def read_img(filename):
	img = cv2.imread(filename)
	return img[:, :, ::-1].astype('float32') / 255.0

def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).copy()

def augment(imgs=[], size=[256,256], only_h_flip=False):
    H, W, _ = imgs[0].shape
    Hc, Wc = size

    # simple re-weight for the edge
    Hs = random.randint(0, H - Hc - 1)

    Ws = random.randint(0, W - Wc - 1)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]
    aug = random.randint(0, 8)
    if aug == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i],0)
    elif aug == 2:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i],1)
    elif aug == 3:
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], 1, (0, 1))
    elif aug == 4:
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], 2, (0, 1))
    elif aug == 5:
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], 3, (0, 1))
    elif aug == 6:
        for i in range(len(imgs)):
            imgs[i] = np.rot90(np.flip(imgs[i], 0), 1, (0, 1))
    elif aug == 7:
        for i in range(len(imgs)):
            imgs[i] = np.rot90(np.flip(imgs[i], 1), 1, (0, 1))

    return imgs


def align(imgs=[], size=256):
    H, W, _ = imgs[0].shape
    Hc, Wc = size

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    return imgs

class PairLoader(Dataset):
    def __init__(self, data_dir, sub_dir, mode, size=[256,256], only_h_flip=False):
        super().__init__()
        self.mode = mode
        self.size = size
        self.only_h_flip = only_h_flip

        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'normal')))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        img_name = self.img_names[idx]
        source_img = read_img(os.path.join(self.root_dir, 'low', img_name.replace('normal','low'))) * 2 - 1
        target_img = read_img(os.path.join(self.root_dir, 'normal', img_name)) * 2 - 1

        if self.mode == 'train':
            [source_img, target_img] = augment([source_img, target_img], self.size,
                                               self.only_h_flip)

        if self.mode == 'valid':
            [source_img, target_img] = align([source_img, target_img], self.size)

        return {'low': hwc_to_chw(source_img), 'normal': hwc_to_chw(target_img), 'filename': img_name}


class SingleLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        img_name = self.img_names[idx]
        img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

        return {'img': hwc_to_chw(img), 'filename': img_name}
