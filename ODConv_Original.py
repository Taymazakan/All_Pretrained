import os
import sys
import torch
import torch.nn as nn

# --------------------------------------------------
# 1. Add ODConv repository to Python path
# --------------------------------------------------
ODCONV_PATH = r"D:\Project\Image processing\ViViT_Ecocardiogram\ViVit\OminiCNN\ODConv"
sys.path.append(ODCONV_PATH)

# --------------------------------------------------
# 2. Import ODConv model
# --------------------------------------------------
from models.od_resnet import od_resnet18


# --------------------------------------------------
# 3. Create ODConv model function
# --------------------------------------------------
def create_odconv_resnet18(num_classes, pretrained_path=None):
    """
    Create ODConv ResNet18 model
    """

    # build backbone
    model = od_resnet18(kernel_num=1, reduction=1/16)

    # replace final classifier
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # --------------------------------------------------
    # 4. Load pretrained weights (optional)
    # --------------------------------------------------
    if pretrained_path is not None and os.path.exists(pretrained_path):

        checkpoint = torch.load(pretrained_path, map_location="cpu")

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        # remove classifier weights (ImageNet = 1000 classes)
        checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("fc.")}

        model.load_state_dict(checkpoint, strict=False)

        print("Pretrained weights loaded")

    return model


# --------------------------------------------------
# 5. Example usage
# --------------------------------------------------
if __name__ == "__main__":

    NUM_CLASSES = 3   # example: AD / MCI / CN or your dataset classes

    model = create_odconv_resnet18(
        num_classes=NUM_CLASSES,
        pretrained_path=None   # put ODConv checkpoint path if available
    )

    print(model)

    # test forward pass
    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print("Output shape:", y.shape)