import torch
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

MAIN_DIR="data/"
SUB_DIR="data/canny/"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100

# transform = A.Compose([
#     A.Resize(width=256, height=256),
#     A.Normalize(mean=[0.5], std=[0.5]),
#     ToTensorV2()
#     ],
#     additional_targets={"converted_image" : "image", }
# )

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(256,256)),
        transforms.Normalize([0.5], [0.5])
    ])