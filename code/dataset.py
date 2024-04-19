import torch.utils.data as data
import PIL.Image as Image
import os
import matplotlib.pyplot as plt

def make_dataset(root):
    imgs = []
    n = len(os.listdir(root)) // 2
    for i in range(n):
        img = os.path.join(root, "%03d.png" % i)
        mask = os.path.join(root, "%03d_mask.png" % i)
        imgs.append((img, mask))
    return imgs


def make_dataset_train():
    imgs = []
    for root, dirs, files in os.walk("../CASIA/data/train/row_data"):
        for file in files:
            print(file)
            #img = file
            root1="../CASIA/data/train/row_data"
            img = os.path.join(root1, file)
            print(img)
            names = file.split('.')[0]
            root2="../CASIA/data/train/mask_data"
            mask = os.path.join(root2, "OperatorA_%s.tiff" % names)
            print(mask)
            imgs.append((img, mask))
    return imgs


def make_dataset_test():
    imgs = []
    for root, dirs, files in os.walk("../CASIA/data/test/row_data"):
        for file in files:
            print(file)
            #img = file
            root1="../CASIA/data/test/row_data"
            img = os.path.join(root1, file)
            print(img)
            names = file.split('.')[0]
            root2="../CASIA/data/test/mask_data"
            mask = os.path.join(root2, "OperatorA_%s.tiff" % names)
            print(mask)
            imgs.append((img, mask))
    return imgs


class LiverDataset_train(data.Dataset):
    def __init__(self,  transform=None, target_transform=None):
        imgs = make_dataset_train()#train
        #imgs=make_dataset_test() #test
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

class LiverDataset_test(data.Dataset):
    def __init__(self,  transform=None, target_transform=None):
        #imgs = make_dataset_train()#train
        imgs=make_dataset_test() #test
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)