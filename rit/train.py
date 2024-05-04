from torch.utils.data import DataLoader
from dataset import PrepareDataset
from config import MAIN_DIR, SUB_DIR, DEVICE, EPOCHS, transform
from utils import show_tensor_image, show_image
from rit_model import DenseNet2D
from model_utils import CrossEntropyLoss2d, GeneralizedDiceLoss
import torch
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Dice
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

writer_gray = SummaryWriter(f"logs/gray")
writer_canny = SummaryWriter(f"logs/canny")

if __name__ == "__main__":
    step = 0
    model = DenseNet2D()
    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr =  1e-3)
    criterion = CrossEntropyLoss2d()
    criterion_DICE = GeneralizedDiceLoss(softmax=True, reduction=True)
    dice = Dice(average="micro").to(DEVICE)

    dataset = PrepareDataset(
        file_path=MAIN_DIR, transform=transform
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(EPOCHS):
        for batch_idx, batchdata in enumerate(loader):
            main_img, target_image, spatialWeights = batchdata
            # show_tensor_image(main_img[0])
   
            main_img = main_img.to(DEVICE)
            target = target_image.to(DEVICE)
            spatialWeights = spatialWeights.to(DEVICE)

            optimizer.zero_grad()
            output_image = model(main_img)
            cross_loss = criterion(output_image, target)
            loss = cross_loss*(torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(DEVICE)+(spatialWeights).to(torch.float32).to(DEVICE))
            loss = torch.mean(loss).to(torch.float32).to(DEVICE)
            loss_dice = dice(output_image,target)
            loss = loss + loss_dice
            loss.backward()
            optimizer.step()
            if batch_idx%10 == 0:
                print('Epoch:{} [{}/{}], Loss: {:.3f}'.format(epoch,batch_idx,len(loader), loss.item()))

                with torch.no_grad():
                    canny_img = model(main_img)

                    canny_pred = torchvision.utils.make_grid(canny_img[:32], normalize=True)
                    gray_img = torchvision.utils.make_grid(main_img[:32], normalize=True)

                    writer_gray.add_image(
                        "gray Images", gray_img, global_step=step
                    )
                    writer_canny.add_image(
                        "canny Images", canny_pred, global_step=step
                    )
                    step += 1