from torch.utils.data import DataLoader
from dataset import PrepareDataset
from config import MAIN_DIR, SUB_DIR, DEVICE, EPOCHS, transform
from utils import show_tensor_image, show_image
from rit_model import DenseNet2D
from model_utils import CrossEntropyLoss2d, GeneralizedDiceLoss
import torch
import pytorch_ssim
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

writer_gray = SummaryWriter(f"logs/gray")
writer_canny = SummaryWriter(f"logs/canny")

if __name__ == "__main__":

    dataset = PrepareDataset(
        root_dir=MAIN_DIR, sub_dir=SUB_DIR, transform=transform
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)