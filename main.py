# main.py 

# 0. import package
import torch
import torch.nn as nn
import os
from transformers import CLIPProcessor, CLIPModel
import time
import json

from fun_tools.functional_tool import load_config                # read .yml configuration files
from fun_tools.functional_tool import get_largest_pth_file       # Load saved .pth model checkpoint 
from model.dataloader import get_dataset, get_dataset_clip, get_dataloader, get_imagenet_classes # Read and load dataset
from model.model_run import train_model_base, train_model_img2clip2, train_model_img2clip  # Import train model
from model.Network_Config import FeatureAlign                    # Import compatible classifier
from model.cifar100_label import cifar100_labels_descriptions

from model.mobileNetV3 import MobileNetV3  
from model.dataloader import load_data     
from model.resNet import resnet18, resnet34, resnet50, resnet101, resnet152  
from model.WRN import WideResNet           

from model.resnetV2 import resnet20, resnet32
from model.wrnV2 import wrn_16_2

torch.backends.cudnn.benchmark = True           # Optimize memory allocation
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent deadlocks via thread configuration

# 1. Read YAML configuration file
# resnet
# model_config = load_config("./config/resnet/1-resnet18-baseline.yml")
# model_config = load_config("./config/resnet/3-resnet34-baseline.yml")
# model_config = load_config("./config/resnet/5-resnet50-baseline.yml")
# model_config = load_config("./config/resnet/7-resnet101-baseline.yml")
# model_config = load_config("./config/resnet/9-resnet152-baseline.yml")

# model_config = load_config("./config/resnet/2-resnet18-classifier.yml")
# model_config = load_config("./config/resnet/4-resnet34-classifier.yml")
# model_config = load_config("./config/resnet/6-resnet50-classifier.yml")
# model_config = load_config("./config/resnet/8-resnet101-classifier.yml")
model_config = load_config("./config/resnet/10-resnet152-classifier.yml")

# model_config = load_config("./config/resnet/20-resnet18-classifier.yml")

# WRN
# model_config = load_config("./config/WRN/1-WRN-16-8-baseline.yml")
# model_config = load_config("./config/WRN/2-WRN-16-8-classifier.yml")
# model_config = load_config("./config/WRN/3-WRN-28-10-baseline.yml")
# model_config = load_config("./config/WRN/4-WRN-28-10-classifier.yml")

# Mobilenet
# model_config = load_config("./config/mobilenet/1-mobilenetv3-baseline.yml")
# model_config = load_config("./config/mobilenet/2-mobilenetv3-classifier.yml")
# model_config = load_config("./config/mobilenet/3-mobilenetv3-baseline.yml")
# model_config = load_config("./config/mobilenet/4-mobilenetv3-classifier.yml")
# model_config = load_config("./config/mobilenet/11-mobilenetv3-classifier.yml")
# print(model_config['model']['model_name'])  

# ablation
# model_config = load_config("./config/abla/1-resnet18-classifier.yml")
# model_config = load_config("./config/abla/2-resnet18-classifier.yml")
# model_config = load_config("./config/abla/3-WRN-16-8-classifier.yml")
# model_config = load_config("./config/abla/4-resnet18-classifier.yml")
# model_config = load_config("./config/abla/5-resnet18-classifier.yml")
# model_config = load_config("./config/abla/6-WRN-16-8-classifier.yml")
# model_config = load_config("./config/abla/7-WRN-16-8-classifier.yml")
# model_config = load_config("./config/abla/8-resnet18-classifier.yml")
# model_config = load_config("./config/abla/9-resnet18-classifier.yml")
# model_config = load_config("./config/abla/10-resnet18-classifier.yml")
# model_config = load_config("./config/abla/11-resnet18-classifier.yml")
# model_config = load_config("./config/abla/12-resnet18-classifier.yml")

# distillation 
# model_config = load_config("./config/distillation/1-resnet20-classifier.yml")
# model_config = load_config("./config/distillation/2-resnet32-classifier.yml")
# model_config = load_config("./config/distillation/3-WRN-16-2-classifier.yml")
# model_config = load_config("./config/distillation/4-MNV2-classifier.yml")

# imagenet
model_config = load_config("./config/ImageNet/1-resnet18-classifier.yml")
# model_config = load_config("./config/ImageNet/2-mobilenetv3-classifier.yml")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# imagenet dataset path
data_root = "/home/user/ImageNet/"
# model_root = "./pretrained/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/"


def load_json_as_dict(file_path):
    with open(file_path, 'r', encoding="gbk") as f:
        data = json.load(f)
    return data


# 1.2 Load cifar100 dataset
model_dataset = get_dataset(dowanload=False, dataset=model_config['dataset']['name'], 
                               root=model_config['dataset']['dataset_loaction'])           # Load both training and test datasets
model_dataset_test = get_dataset(is_train=False, dowanload=False, dataset=model_config['dataset']['name'], 
                               root=model_config['dataset']['dataset_loaction'], is_transform="test")
dataset_classes_name, model_dataloader = get_dataloader(model_dataset, batch_size=model_config['dataset']['batch_size'])
_, model_dataloader_test = get_dataloader(model_dataset_test, batch_size=model_config['dataset']['batch_size'])

# 1.2 Load imagenet dataset
# model_dataloader, model_dataloader_test = load_data(dowanload=False, dataset="IMAGENET",
#                                                     root=data_root, batch_size=model_config['dataset']['batch_size'])
# dataset_classes_name_index = get_imagenet_classes(data_root)
# file_path = './model/imagenet_classes.json'
# dataset_classes_name = load_json_as_dict(file_path)
print("--------Successfully loaded dataset--------")

# 1.1 Create corresponding model based on config
if model_config['model']['model_name'][:6] == "resnet":          # Determine model type
    if model_config['model']['model_name'].split("-")[1] == "18":
        neur_model = resnet18(num_classes=model_config['dataset']['num_classes']).to(device)
    elif model_config['model']['model_name'].split("-")[1] == "34":
        neur_model = resnet34(num_classes=model_config['dataset']['num_classes']).to(device)
    elif model_config['model']['model_name'].split("-")[1] == "50":
        neur_model = resnet50(num_classes=model_config['dataset']['num_classes']).to(device)
    elif model_config['model']['model_name'].split("-")[1] == "101":
        neur_model = resnet101(num_classes=model_config['dataset']['num_classes']).to(device)
    elif model_config['model']['model_name'].split("-")[1] == "152":
        neur_model = resnet152(num_classes=model_config['dataset']['num_classes']).to(device)
    else:
        print("ResNet config file error detected........")    

elif model_config['model']['model_name'][:6] == "(CLIP)":
    print("-------- Loading CLIP model------")
    time1 = time.time()
    # load clip model 
    clip_model = CLIPModel.from_pretrained(model_config['model']['clip_model']).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_config['model']['clip_model'])
    print("--------Time elapsed for CLIP model loading------    {:.4f}mins".format((time.time()-time1)/60))
    # Freeze CLIP model parameters (no updates)
    for param in clip_model.parameters():
        param.requires_grad = False
    
    # Verify network architecture
    if model_config['model']['model_name'][6:12] == "resnet":
        if model_config['model']['model_name'].split("-")[1] == "18":
            featureAlign = FeatureAlign(input_dim=256)
            # featureAlign = FeatureAlign(input_dim=64)
            neur_model = resnet18(num_classes=model_config['dataset']['num_classes']).to(device)
        elif model_config['model']['model_name'].split("-")[1] == "34":
            featureAlign = FeatureAlign(input_dim=256)
            neur_model = resnet34(num_classes=model_config['dataset']['num_classes']).to(device)
        elif model_config['model']['model_name'].split("-")[1] == "50":
            featureAlign = FeatureAlign(input_dim=1024)
            neur_model = resnet50(num_classes=model_config['dataset']['num_classes']).to(device)
        elif model_config['model']['model_name'].split("-")[1] == "101":
            featureAlign = FeatureAlign(input_dim=1024)
            neur_model = resnet101(num_classes=model_config['dataset']['num_classes']).to(device)
        elif model_config['model']['model_name'].split("-")[1] == "152":
            featureAlign = FeatureAlign(input_dim=1024)
            neur_model = resnet152(num_classes=model_config['dataset']['num_classes']).to(device)
        else:
            print("ResNet config file error detected...........")   
        neur_model_temp =  neur_model

    elif model_config['model']['model_name'][6:14] == "ResnetV2":
        if model_config['model']['model_name'].split("-")[1] == "20":
            # featureAlign = FeatureAlign(input_dim=32)
            featureAlign = FeatureAlign(input_dim=256)
            neur_model = resnet20(num_classes=model_config['dataset']['num_classes']).to(device)
        elif model_config['model']['model_name'].split("-")[1] == "32":
            # featureAlign = FeatureAlign(input_dim=32)
            featureAlign = FeatureAlign(input_dim=256)
            neur_model = resnet32(num_classes=model_config['dataset']['num_classes']).to(device)
        neur_model_temp =  neur_model
    
    elif model_config['model']['model_name'][6:9] == "wrn":
        featureAlign = FeatureAlign(input_dim=512)
        neur_model_temp = wrn_16_2(num_classes=model_config['dataset']['num_classes']).to(device)
    
    # WRN
    elif model_config['model']['model_name'][6:9] == "WRN":
        wrn_depth = model_config['model']['model_name'].split("-")[1]
        wrn_factor = model_config['model']['model_name'].split("-")[2]
        if wrn_depth == "16":
            featureAlign = FeatureAlign(input_dim=256)
            # featureAlign = FeatureAlign(input_dim=128)
        elif wrn_depth == "28":
            featureAlign = FeatureAlign(input_dim=320)
        neur_model_temp = WideResNet(depth=int(wrn_depth) , width_factor=int(wrn_factor), 
                                     num_classes=model_config['dataset']['num_classes']).to(device)

    # mobilenet
    elif model_config['model']['model_name'][6:15] == "mobilenet":
        
        model_dataloader, model_dataloader_test = load_data(dowanload=False, dataset=model_config['dataset']['name'],
                                                    root=model_config['dataset']['dataset_loaction'], batch_size=model_config['dataset']['batch_size'])
        if model_config['model']['model_name'].split("-")[1] == "SMALL":
            featureAlign = FeatureAlign(input_dim=672)
        elif model_config['model']['model_name'].split("-")[1] == "LARGE":
            featureAlign = FeatureAlign(input_dim=1120)     # 960
        neur_model_temp = MobileNetV3(model_mode=model_config['model']['model_name'].split("-")[1], num_classes=model_config['dataset']['num_classes'], multiplier=1.0, dropout_rate=0.0).to(device)  

# WRN baseline
elif model_config['model']['model_name'][:3] == "WRN":
    wrn_depth = model_config['model']['model_name'].split("-")[1]
    wrn_factor = model_config['model']['model_name'].split("-")[2]
    neur_model = WideResNet(depth=int(wrn_depth) , width_factor=int(wrn_factor), num_classes=model_config['dataset']['num_classes'])
    neur_model = neur_model.to(device)

# mobilenet baseline
elif model_config['model']['model_name'][:9] == "mobilenet":
    model_dataloader, model_dataloader_test = load_data(dowanload=False, dataset=model_config['dataset']['name'],
                                                    root=model_config['dataset']['dataset_loaction'], batch_size=model_config['dataset']['batch_size'])

    neur_model = MobileNetV3(model_mode=model_config['model']['model_name'].split("-")[1], num_classes=model_config['dataset']['num_classes'], multiplier=1.0, dropout_rate=0.0).to(device)  
    
print("Running: ", model_config['model']['group_name'], " Configuration file")


# 1.3 Check if saved model exists -> Load if available 
save_path = os.path.join(model_config['save_model_pth']['save_model_pth_location'], 
                               model_config['model']['model_name'], "save")

# 2. Start training and save checkpoints  
print("-------Start Training---")
if "base" in model_config['model']['group_name']:

    if os.path.exists(save_path):
        path = get_largest_pth_file(save_path, model_config['model']['group_name'])
        neur_model.load_state_dict(torch.load(path))
        print("-------Successfully loaded saved model ---", path)

    # Train baseline model
    train_model_base(model_config['model']['num_epochs'], neur_model, model_dataloader, save_pth=model_config['save_train_log']['train_train_location'],
                 name=model_config['model']['model_name'], group_name=model_config['model']['group_name'], test_dataloader=model_dataloader_test)
elif "CLIP" in model_config['model']['group_name']:

    if os.path.exists(save_path):
        # loda model
        path = get_largest_pth_file(save_path, model_config['model']['group_name']+"_(CLIP)")
        neur_model_temp.load_state_dict(torch.load(path ))
        # load FAdapter
        path2 = get_largest_pth_file(save_path, model_config['model']['group_name']+"_(featue")
        featureAlign.load_state_dict(torch.load(path2))
        print("-------Successfully loaded saved model and FApater----")


    # train_model_img2clip2(model_config['model']['num_epochs'], clip_model, clip_processor, neur_model_temp, featureAlign, model_dataloader,
    #                      dataset_classes_name, model_dataloader_test, save_path=model_config['save_train_log']['train_train_location'], 
    #                      name=model_config['model']['model_name'], group_name=model_config['model']['group_name'])
    # CFD-CLIP
    train_model_img2clip(model_config['model']['num_epochs'], clip_model, clip_processor, neur_model_temp, featureAlign, model_dataloader,
                         dataset_classes_name, model_dataloader_test, save_path=model_config['save_train_log']['train_train_location'], 
                         name=model_config['model']['model_name'], group_name=model_config['model']['group_name'])
