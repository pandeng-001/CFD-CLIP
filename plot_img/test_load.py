# test Load pre-trained model
import os
from fun_tools.functional_tool import get_largest_pth_file
from model.resNet import resnet18, resnet34, resnet50, resnet101
from model.WRN import WideResNet
from model.model_run import evaluate_model_base
import torch
from model.Network_Config import FeatureAlign 
from model.dataloader import get_dataset,  get_dataloader  

model_A = resnet18(num_classes=100)
featureAlign_A = FeatureAlign(input_dim=256)  # resnet50 1024  resnet34 256 

name = "resnet18"
save_path = os.path.join("./log", "resnet-18", "save")  # 2- resnet18
save_path2 = os.path.join("./log", "resnet-101", "save")  # 4- resnet34

# save_feaA_path = os.path.join("./log", "(CLIP)resnet-101", "save")

if os.path.exists(save_path):
    path1 = get_largest_pth_file(save_path, "1-resnet18-")  
    path2 = get_largest_pth_file(save_path, "3-WRN-28-10-")  

    path3 = get_largest_pth_file(save_path, f"2-(CLIP)WRN-16-8-classifier"+"_(featue")  
    path4 = get_largest_pth_file(save_path, f"4-(CLIP)WRN-28-10-classifier"+"_(featue")  
    print(path1)
    print(path2)
    model_A.load_state_dict(torch.load(path1))
    # featureAlign_A.load_state_dict(torch.load(path3))
    # featureAlign.load_state_dict(torch.load(path2))
    print("-------Successfully loaded saved models---", path1, path2)

# 2 load dataset
model_dataset_test = get_dataset(is_train=False, dowanload=False, dataset="cifar100", 
                               root="./data", is_transform="test")
dataloader_class_name, model_dataloader_test = get_dataloader(model_dataset_test, batch_size=512)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_A = model_A.to(device)
featureAlign_A = featureAlign_A.to(device)

top1_acc, top5_acc = evaluate_model_base(model_A, model_dataloader_test)
print("top1_acc: ",top1_acc,"top5_acc: ",top5_acc)
