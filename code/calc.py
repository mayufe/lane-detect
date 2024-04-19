from torchvision.models import resnet50
from unet44 import Unet
from thop import profile
from thop import clever_format
import torch

model=Unet(3,1)
input = torch.randn(1, 3, 512, 512)

macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")

print(macs)
print(params)
