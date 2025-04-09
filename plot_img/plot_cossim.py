# 0 import package 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import os
from fun_tools.functional_tool import get_largest_pth_file
from model.resNet import resnet18, resnet34, resnet50, resnet101
from model.WRN import WideResNet

import torch
from model.Network_Config import FeatureAlign 
from model.dataloader import get_dataset,  get_dataloader  
import torchvision

# 1 Load pre-trained model

# Load two models 
model_A = resnet50(num_classes=100)
model_B = resnet34(num_classes=100)
# model = resnet50(num_classes=100)
# model = resnet101(num_classes=100)
model_C = WideResNet(depth=16, width_factor=8, num_classes=100)
model_D = WideResNet(depth=28, width_factor=10, num_classes=100)

featureAlign = FeatureAlign(input_dim=1024)  # resnet50 1024  resnet34 256 
name = "resnet18"
save_path = os.path.join("./log", "resnet-50", "save")  # 1- resnet18
save_path2 = os.path.join("./log", "resnet-34", "save")  # 3- resnet34
save_path3 = os.path.join("./log", "WRN-16-8", "save")  # 3- resnet34
save_path4 = os.path.join("./log", "WRN-28-10", "save")  # 3- resnet34

# save_path = os.path.join("./log", "resnet-50", "save")    # 5- resnet50
# save_path = os.path.join("./log", "resnet-101", "save")  # 7- resnet101
# save_feaA_path = os.path.join("./log", "(CLIP)resnet-101", "save")

if os.path.exists(save_path):
    path1 = get_largest_pth_file(save_path, "5-resnet50-")  
    path2 = get_largest_pth_file(save_path2, "3-resnet34-")  
    path3 = get_largest_pth_file(save_path3, "1-WRN-")  
    path4 = get_largest_pth_file(save_path4, "3-WRN-")  

    # path2 = get_largest_pth_file(save_feaA_path, f"6-(CLIP){name}-classifier"+"_(featue")  # 特征对齐
    print(path1)
    print(path2)
    model_A.load_state_dict(torch.load(path1))
    model_B.load_state_dict(torch.load(path2))
    model_C.load_state_dict(torch.load(path3))
    model_D.load_state_dict(torch.load(path4))
    # featureAlign.load_state_dict(torch.load(path2))
    print("-------Successfully loaded saved models---", path1, path2)

# 2 Load dataset
model_dataset_test = get_dataset(is_train=False, dowanload=False, dataset="cifar100", 
                               root="./data", is_transform="test")
_, model_dataloader_test = get_dataloader(model_dataset_test, batch_size=512)

#  model is a PyTorch model (e.g., ResNet)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_A = model_A.to(device)
model_B = model_B.to(device)
model_C = model_C.to(device)
model_D = model_D.to(device)
featureAlign = featureAlign.to(device)
model_A.eval()
model_B.eval()
model_C.eval()
model_D.eval()

# Calculate per-batch cosine similarity
avg_cos_sim = []
final_avg_cos_sim_list = []

with torch.no_grad():
    for images, labels in model_dataloader_test:
        images = images.to(device)
        labels = labels.to(device)

        A, _ = model_A(images)
        B, _ = model_B(images)
        C, _ = model_C(images) 
        D, _ = model_D(images) 
        A = torch.mean(A, dim=[2, 3])  # Average over width and height dimensions
        B = torch.mean(B, dim=[2, 3])  # Average over width and height dimensions
        C = torch.mean(C, dim=[2, 3])
        D = torch.mean(D, dim=[2, 3])

        D = D[:,:256]
        print(A.shape, B.shape)
        # Compute cosine similarity matrix [batch_size, batch_size]
        cos_sim = cosine_similarity(C.cpu().numpy(), D.cpu().numpy())
        # Calculate average cosine similarity per batch
        avg_cos_sim.append(np.max(cos_sim))

# Compute overall mean cosine similarity across all batches
final_avg_cos_sim = np.mean(avg_cos_sim)
final_avg_cos_sim_list.append(final_avg_cos_sim)
print(final_avg_cos_sim_list)

# Plot bar chart
plt.figure(figsize=(10, 6))
plt.bar(len(final_avg_cos_sim_list), final_avg_cos_sim, color='skyblue', label='Batch Average Cosine Similarity')
plt.xlabel('Batch Number')
plt.ylabel('Average Cosine Similarity')
plt.title('Average Cosine Similarity Between A and B Features Across Batches')
plt.legend()  # Display legend
plt.show()
plt.savefig("./plot_img/img/resnet18_cossim.png")
