import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import os.path as osp
import random
from torchvision import transforms
import cv2

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(256,256)),
        transforms.Normalize([0.5], [0.5])
    ])

class RandomHorizontalFlip(object):
    def __call__(self, img, label):
        if random.random() < 0.5:
            return np.fliplr(img), np.fliplr(label)
        return img,label

class PrepareDataset(Dataset):
    def __init__(self, file_path, split='train', transform=None):
        super().__init__()
        self.transform = transform
        self.filepath= os.path.join(file_path, split)
        self.split = split
        listall = []

        for file in os.listdir(os.path.join(self.filepath,'images')):   
            if file.endswith(".jpg"):
               listall.append(file.strip(".jpg"))
        self.list_files=listall

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, idx):
        imagepath = os.path.join(self.filepath,'images',self.list_files[idx]+'.jpg')
        pil_img = Image.open(imagepath).convert("L")
        # H, W = pil_img.width , pil_img.height

        if self.split != 'test':
            labelpath = os.path.join(self.filepath,'images',self.list_files[idx]+'.jpg')
            label = Image.open(labelpath).convert("L")    
            label = np.array(label.resize((256, 256)))  # Ensure resizing happens on numpy array
            label = cv2.Canny(label, 10, 20) / 255
            
        
        img = np.array(pil_img)
        if self.transform is not None:
            if self.split == 'train':
                img, label = RandomHorizontalFlip()(img,label)
            img = np.array(img, dtype=np.float32)
            img = self.transform(img)
        
        label = MaskToTensor()(label)
        return img, label

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()       
 
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = PrepareDataset("data/", split="train", transform=transform)
    img, label = dataset[0]
    plt.subplot(121)
    plt.imshow(np.array(label))
    plt.subplot(122)
    plt.imshow(np.array(img)[0,:,:],cmap='gray')
    plt.show()
