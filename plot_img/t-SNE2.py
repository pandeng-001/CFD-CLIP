# vanill resnet   t-sne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import os
from fun_tools.functional_tool import get_largest_pth_file
from model.resNet import resnet18, resnet34, resnet50, resnet101
import torch
from model.Network_Config import FeatureAlign  
from model.dataloader import get_dataset,  get_dataloader  
import torchvision

# load dataset
model_dataset_test = get_dataset(is_train=False, dowanload=False, dataset="cifar100", 
                               root="./data", is_transform="test")
_, model_dataloader_test = get_dataloader(model_dataset_test, batch_size=512)


# load model
model = resnet18(num_classes=100)
# model = resnet34(num_classes=100)
# model = resnet50(num_classes=100)
# model = resnet101(num_classes=100)
featureAlign = FeatureAlign(input_dim=1024)  # resnet50 1024  resnet34 256 

name = "resnet18"
save_path = os.path.join("./log", "resnet-18", "save")  # 1- resnet18
# save_path = os.path.join("./log", "resnet-34", "save")  # 3- resnet34
# save_path = os.path.join("./log", "resnet-50", "save")    # 5- resnet50
# save_path = os.path.join("./log", "resnet-101", "save")  # 7- resnet101

# save_feaA_path = os.path.join("./log", "(CLIP)resnet-101", "save")
if os.path.exists(save_path):
    path1 = get_largest_pth_file(save_path, f"1-{name}-")  
    # path2 = get_largest_pth_file(save_feaA_path, f"6-(CLIP){name}-classifier"+"_(featue")  # 特征对齐
    print(path1)
    # print(path2)
    model.load_state_dict(torch.load(path1))
    # featureAlign.load_state_dict(torch.load(path2))
    print("-------Successfully loaded saved models---", path1)


# Extract features and class labels
def TSNE_visualization(features, labels):
    # 1. Normalize feature data
    features = StandardScaler().fit_transform(features)

    # 2. Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=80, max_iter=1000)
    tsne_results = tsne.fit_transform(features)

    # 3. Return reduced data with labels  
    return tsne_results[:, 0], tsne_results[:, 1], labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
featureAlign = featureAlign.to(device)
model.eval()

all_features = []
all_outputs = []
all_labels = []

with torch.no_grad():
    for images, labels in model_dataloader_test:
        images = images.to(device)
        labels = labels.to(device)
        
        features, _ = model(images)
        # features = features.view(features.size(0), -1)  
        features = torch.mean(features, dim=[2, 3])  
        # features = featureAlign(features)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Concatenate all batches 
all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)


# 使用 t-SNE 降维
x_tsne, y_tsne, labels = TSNE_visualization(all_features, all_labels)


selected_classes = [i for i in range(50)]  # Filter specified classes 
mask = np.isin(labels, selected_classes)

x_selected = x_tsne[mask]
y_selected = y_tsne[mask]
labels_selected = labels[mask]

# plot t-SNE 
plt.figure(figsize=(8, 8))
scatter = plt.scatter(x_selected, y_selected, c=labels_selected, cmap="tab10", s=2, alpha=0.9)

# plt.colorbar(scatter, label="Classes")
# plt.title("t-SNE Visualization of CIFAR-100 Features")
# plt.xlabel("t-SNE Dim 1")
# plt.ylabel("t-SNE Dim 2")
# Remove axes  
# Customize borders:  
# - Keep frame but remove ticks  
ax = plt.gca()  
ax.spines['top'].set_visible(True)       #  - Show top border  
ax.spines['right'].set_visible(True)     #  - Show right border  
ax.spines['bottom'].set_visible(True)    #  - Show bottom border  
ax.spines['left'].set_visible(True)      #  - Show left border  

# Remove ticks 
ax.set_xticks([])  # Hide x-axis ticks  
ax.set_yticks([])  # Hide y-axis ticks  
plt.savefig("./plot_img/img/resnet18(van)_cla_tsne.png")
# plt.savefig("./plot_img/img/resnet34(van)_cla_tsne.png")
# plt.savefig("./plot_img/img/resnet50(van)_cla_tsne.png")
# plt.savefig("./plot_img/img/resnet101(van)_cla_tsne.png")
# plt.show()