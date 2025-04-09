# 0 import package 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel

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
model_B = resnet101(num_classes=100)
model_C = WideResNet(depth=16, width_factor=8, num_classes=100)
model_D = WideResNet(depth=28, width_factor=10, num_classes=100)


featureAlign_A = FeatureAlign(input_dim=1024)  # resnet50 1024  resnet34 256 
featureAlign_B = FeatureAlign(input_dim=1024)  
featureAlign_C = FeatureAlign(input_dim=256)
featureAlign_D = FeatureAlign(input_dim=320)
name = "resnet18"
save_path = os.path.join("./log", "resnet-50", "save")  # 2- resnet18
save_path2 = os.path.join("./log", "resnet-101", "save")  # 4- resnet34
save_path3 = os.path.join("./log", "WRN-16-8", "save")  
save_path4 = os.path.join("./log", "WRN-28-10", "save")  

save_path5 = os.path.join("./log", "(CLIP)WRN-16-8", "save")  # 2- resnet18
save_path6 = os.path.join("./log", "(CLIP)WRN-28-10", "save")  # 4- resnet34

# save_path = os.path.join("./log", "resnet-50", "save")    # 5- resnet50
# save_path = os.path.join("./log", "resnet-101", "save")  # 7- resnet101
# save_feaA_path = os.path.join("./log", "(CLIP)resnet-101", "save")

if os.path.exists(save_path):
    path1 = get_largest_pth_file(save_path3, "1-WRN-16-8-")  
    path2 = get_largest_pth_file(save_path4, "3-WRN-28-10-")  

    path3 = get_largest_pth_file(save_path5, f"2-(CLIP)WRN-16-8-classifier"+"_(featue")   # faeture align
    path4 = get_largest_pth_file(save_path6, f"4-(CLIP)WRN-28-10-classifier"+"_(featue")  # faeture align
    print(path1)
    print(path2)
    model_C.load_state_dict(torch.load(path1))
    model_D.load_state_dict(torch.load(path2))
    featureAlign_C.load_state_dict(torch.load(path3))
    featureAlign_D.load_state_dict(torch.load(path4))
    # featureAlign.load_state_dict(torch.load(path2))
    print("-------Successfully loaded saved models---", path1, path2)

# 2 Load dataset
model_dataset_test = get_dataset(is_train=False, dowanload=False, dataset="cifar100", 
                               root="./data", is_transform="test")
dataloader_class_name, model_dataloader_test = get_dataloader(model_dataset_test, batch_size=512)

# model is a PyTorch model (e.g., ResNet)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model_A = model_A.to(device)
model_B = model_B.to(device)
model_C = model_C.to(device)
model_D = model_D.to(device)
featureAlign_A = featureAlign_A.to(device)
featureAlign_B = featureAlign_B.to(device)
featureAlign_C = featureAlign_C.to(device)
featureAlign_D = featureAlign_D.to(device)
model_A.eval()
model_B.eval()
model_C.eval()
model_D.eval()

# Calculate per-batch cosine similarity
avg_cos_sim = []
final_avg_cos_sim_list = []

avg_cos_sim2 = []
final_avg_cos_sim_list2 = []

with torch.no_grad():
    for images, labels in model_dataloader_test:
        images = images.to(device)
        labels = labels.to(device)
        text = [f"a photo of the {dataloader_class_name[i]}"for i in labels]
        clip_inputs = clip_processor(text=text, images=images, return_tensors="pt", padding=True, do_rescale=False)
        clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
        clip_outputs = clip_model(**clip_inputs)
        clip_img_embed = clip_outputs.image_embeds      
        clip_text_embed = clip_outputs.image_embeds     

        A, _ = model_C(images)
        B, _ = model_D(images)
        # A = torch.mean(A, dim=[2, 3])  
        # B = torch.mean(B, dim=[2, 3])  
        # print(A.shape, B.shape)
        A = featureAlign_C(A)
        B = featureAlign_D(B)
        # Compute cosine similarity matrix [batch_size, batch_size]
        cos_sim = cosine_similarity(A.cpu().numpy(), clip_img_embed.cpu().numpy())
        # Calculate average cosine similarity per batch
        avg_cos_sim.append(np.max(cos_sim))

        cos_sim2 = cosine_similarity(clip_text_embed.cpu().numpy(), B.cpu().numpy())
        avg_cos_sim2.append(np.max(cos_sim2))
        # print("clip: ",np.max(cosine_similarity(clip_img_embed.cpu().numpy(), clip_text_embed.cpu().numpy())))


#  Compute overall mean cosine similarity across all batches
final_avg_cos_sim = np.max(avg_cos_sim)
final_avg_cos_sim2 = np.max(avg_cos_sim2)
# print(avg_cos_sim)
final_avg_cos_sim_list.append(final_avg_cos_sim)
final_avg_cos_sim_list2.append(final_avg_cos_sim2)

print("img: ",final_avg_cos_sim_list)
print("txt: ",final_avg_cos_sim_list2)
# Plot bar chart
plt.figure(figsize=(10, 6))
plt.bar(len(final_avg_cos_sim_list), final_avg_cos_sim, color='skyblue', label='Batch Average Cosine Similarity')
plt.xlabel('Batch Number')
plt.ylabel('Average Cosine Similarity')
plt.title('Average Cosine Similarity Between A and B Features Across Batches')
plt.legend()  # Display legend
plt.show()
plt.savefig("./plot_img/img/resnet18_cossim.png")
