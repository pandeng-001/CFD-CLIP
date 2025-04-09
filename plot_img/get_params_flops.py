from ptflops import get_model_complexity_info
# from torchvision.models import resnet18

import os
from fun_tools.functional_tool import get_largest_pth_file
from model.resNet import resnet18, resnet34, resnet50, resnet101, resnet152
from model.WRN import WideResNet  
from model.mobileNetV3 import MobileNetV3  
import torch
from model.Network_Config import FeatureAlign  
from model.dataloader import get_dataset,  get_dataloader 
import torchvision

from model.resnetV2 import resnet20, resnet32  

# resnet 
# model = resnet18(num_classes=100)
# model = resnet34(num_classes=100)
# model = resnet50(num_classes=100)
# model = resnet101(num_classes=100)
# model = resnet152(num_classes=100)
# model = resnet18()

model = resnet20(num_classes=100)

# WRN
# model = WideResNet(depth=16, width_factor=8, num_classes=100)
# model = WideResNet(depth=28, width_factor=10, num_classes=100)

# mobilenetV3
# model = MobileNetV3(model_mode="SMALL", num_classes=100, multiplier=1.0, dropout_rate=0.0)  
# model = MobileNetV3(model_mode="LARGE", num_classes=100, multiplier=1.0, dropout_rate=0.0)  


flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True)
print(f"FLOPs: {flops}, Params: {params}")

featureAlign = FeatureAlign(input_dim=256)
flops, params = get_model_complexity_info(featureAlign, (256, 8, 8), as_strings=True)
print(f"FLOPs: {flops}, Params: {params}")