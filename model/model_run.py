# run model 

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time
from fun_tools.functional_tool import plot_img, save_txt
import sys
sys.path.append("..")
from model.Network_Config import contrastive_loss, ConvReducer, combined_model_classifier, FeatureAlign

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Tests classification accuracy of baseline mode
def evaluate_model_base(model, data_loader):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    top5_correct = 0 
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, output = model(inputs)
            correct += (output.argmax(1) == labels).sum().item()
            top5_values, top5_indices = output.topk(5, dim=1, largest=True, sorted=True)
            top5_correct += sum([labels[i].item() in top5_indices[i].tolist() for i in range(len(labels))])

        print("classification accuracy: ",  correct/len(data_loader.dataset))
        return correct/len(data_loader.dataset), top5_correct/len(data_loader.dataset)


# Evaluate accuracy of CLIP model + classifier  
def evaluate_model_classifier(classifier, clip_model, clip_processor, test_dataloader, classes_name):
    classifier.eval()
    # Count correct predictions
    correct = 0
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad(): 
        for orig_input, label in test_dataloader:
            text = [f"a photo of the {classes_name[i]}"for i in label]
            orig_input, label = orig_input.to(device), label.to(device)
            clip_inputs = clip_processor(text=text, images=orig_input, return_tensors="pt", padding=True, do_rescale=False)
            # Move each tensor in clip_inputs to GPU device
            clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
            clip_outputs = clip_model(**clip_inputs)
            clip_img_embed = clip_outputs.image_embeds        # Extract image embedding features
            outputs = classifier(clip_img_embed)
            correct += (outputs.argmax(1) == label).sum().item()

    return correct/len(test_dataloader.dataset)



# Train baseline model and evaluate accuracy
def train_model_base(num_epochs, model, dataloader, save_pth="./log", name="model", group_name='base', test_dataloader=None):

    crition = nn.CrossEntropyLoss() 
    reset_optimizer = optim.SGD(model.parameters(), lr=0.1,  momentum=0.9, weight_decay=1e-4)      # Add parameter regularization (weight decay)
    # reset_optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(reset_optimizer, T_max=num_epochs)
    trian_model_losses = []     # Track model loss
    train_accuracies = []       # Monitor training accurac
    test_accuracies  = []       # Evaluate test accuracy
    spend_times = []            # Save training time per epoch
    
    for epoch_index in range(num_epochs):
        start_t4 = time.time()

        model.train()
        # Record: model loss, correct predictions count
        total_resnet_loss, correct = 0, 0 
        for orig_input, label in tqdm(dataloader, desc=f"Epoch {epoch_index+1}", unit="batch"):
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            orig_input, label = orig_input.to(device), label.to(device)
            _, reset_resnet18_outputs = model(orig_input)

            # Compute total loss
            reset_loss = crition(reset_resnet18_outputs, label)

            print(f"Epoch {epoch_index+1} loss:",  reset_loss.item())
            reset_optimizer.zero_grad()   # Zero out gradients
            reset_loss.backward()         # Update model parameters
            reset_optimizer.step()        # Apply gradient updates

            total_resnet_loss += reset_loss.item()
            correct += (reset_resnet18_outputs.argmax(1) == label).sum().item()

            if epoch_index%20 == 0:
                # Create directory if not exists
                os.makedirs(os.path.join(save_pth, name, "save"), exist_ok=True)
                torch.save(model.state_dict(), "{}/{}/save/{}_base_{}.pth".format(save_pth, name, group_name, epoch_index))
                torch.save(reset_optimizer.state_dict(), "{}/{}/save/{}_(optimizer){}_{}.pth".format(save_pth, name, group_name, name, epoch_index))
                torch.save(scheduler.state_dict(), "{}/{}/save/{}_(scheduler){}_{}.pth".format(save_pth, name, group_name, name, epoch_index))
        
        scheduler.step()   # update learning rate

        trian_resnet_loss, train_accuracy =  total_resnet_loss/len(dataloader), correct/len(dataloader.dataset)
        trian_model_losses.append(trian_resnet_loss)    
        train_accuracies.append(train_accuracy)  
        test_accuracy = evaluate_model_base(model, test_dataloader)
        test_accuracies.append(test_accuracy)
        # print(trian_resnet_loss, train_accuracy, test_accuracy)
        print(f"Epoch{epoch_index+1}/{num_epochs}, Resnet_Loss: {trian_resnet_loss:.4f},",
                    f"Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy[0]:.4f}")
        print("Time elapsed for one epoch: ", (time.time()-start_t4)/60)
        spend_times.append((time.time()-start_t4)/60)
    # Initialize save directory
    os.makedirs(os.path.join(save_pth, name, "img"), exist_ok=True)
    os.makedirs(os.path.join(save_pth, name, "txt"), exist_ok=True)
    plot_img("Epoch", "Loss", "Training Loss Curve", "{}/{}/img/{}_training_loss_curve.png".format(save_pth, name, group_name),
                                            [trian_model_losses, "Train Total Loss", "red"])
    plot_img("Epoch", "Accuracies", "Accuracies Curve", "{}/{}/img/{}_training_and_test_accuracies_curve.png".format(save_pth, name, group_name), 
                                    [train_accuracies, "Train Accuracies", "red"], [test_accuracies, "Test Accuracies", "yellow"])


    # Log training metrics
    save_txt(trian_model_losses, "{}/{}/txt/{}__trian_losses.txt".format(save_pth, name, group_name))
    save_txt(train_accuracies, "{}/{}/txt/{}_train_accuracies.txt".format(save_pth, name, group_name))
    save_txt(test_accuracies, "{}/{}/txt/{}__test_accuracies.txt".format(save_pth, name, group_name))
    save_txt(spend_times, "{}/{}/txt/{}__time_mins.txt".format(save_pth, name, group_name))


# Train CLIP + classifier model and evaluate accuracy
def train_clip_classifier(num_epochs, clip_model, clip_processor, classifier, dataloader, classes_name, save_path="./log", 
                          name="clip_classifier", group_name='base', labels_desc=None, test_dataloader=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crition = nn.CrossEntropyLoss()
    new_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    trian_model_losses = []       # Track model loss
    train_accuracies = []         # Monitor training accuracy
    test_accuracies = []          # Evaluate test accuracy

    for epoch_index in range(num_epochs):
        start_t4 = time.time()
        classifier.train()
        # Record: model loss, correct predictions count
        total_classifier_loss, correct = 0, 0 
        for orig_input, label in tqdm(dataloader, desc=f"Epoch {epoch_index+1}", unit="batch"):
            if not labels_desc:
                text = [f"a photo of the {classes_name[i]}"for i in label]
            else:
                text = [labels_desc[classes_name[i]] for i in label]
            orig_input, label = orig_input.to(device), label.to(device)
            clip_inputs = clip_processor(text=text, images=orig_input, return_tensors="pt", padding=True, do_rescale=False)
            clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
            clip_outputs = clip_model(**clip_inputs)
            clip_img_embed = clip_outputs.image_embeds  

            outputs = classifier(clip_img_embed)
            loss = crition(outputs, label)
            print(f"第{epoch_index+1}次损失为:",  loss.item())
            new_optimizer.zero_grad()
            loss.backward()              # Zero out gradients
            new_optimizer.step()         # Update model parameters
            total_classifier_loss += loss.item()
            correct += (outputs.argmax(1) == label).sum().item()

            if epoch_index%10 == 0:
                # Create directory if not exists
                os.makedirs(os.path.join(save_path, name, "save"), exist_ok=True)
                torch.save(classifier.state_dict(), "{}/save/{}_{}.pth".format(save_path, name, epoch_index))
        
        
        trian_model_loss, train_accuracy = total_classifier_loss/len(dataloader), correct/len(dataloader.dataset)
        trian_model_losses.append(trian_model_loss)    
        train_accuracies.append(train_accuracy)  
        test_accuracy = evaluate_model_classifier(classifier, clip_model, clip_processor, test_dataloader, classes_name)       # 测试
        test_accuracies.append(test_accuracy)
        print(f"Epoch{epoch_index+1}/{num_epochs}, Resnet_Loss: {trian_model_loss:.4f},",
                    f"Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
        print("Time elapsed for one epoch: ", (time.time()-start_t4)/60)
    
    os.makedirs(os.path.join(save_path, name, "img"), exist_ok=True)
    os.makedirs(os.path.join(save_path, name, "txt"), exist_ok=True)
    plot_img("Epoch", "Loss", "Training Loss Curve", "{}/{}/img/training_loss_curve.png".format(save_path, name),
                                            [trian_model_losses, "Train Total Loss", "red"])

    plot_img("Epoch", "Accuracies", "Accuracies Curve", "{}/{}/img/training_and_test_accuracies_curve.png".format(save_path, name),
                                    [train_accuracies, "Train Accuracies", "red"], [test_accuracies, "Test Accuracies", "yellow"])

    # Log training metrics
    save_txt(trian_model_losses, "{}/{}/txt/trian_{}_losses.txt".format(save_path, name, name))
    save_txt(train_accuracies, "{}/{}/txt/train_{}_accuracies.txt".format(save_path, name, name))
    save_txt(test_accuracies, "{}/{}/txt/test_{}_accuracies.txt".format(save_path, name, name))


#  Train CLIP / classifier model without accuracy choose
def train_model_img2clip2(num_epochs, clip_model, clip_processor, model, featueAlign, dataloader, classes_name, test_dataloader=None, save_path="./log", 
                          name="clip_model", group_name='base', labels_desc=None,):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crition = nn.CrossEntropyLoss() 
    reset_optimizer = optim.SGD(model.parameters(), lr=0.1,  momentum=0.9, weight_decay=1e-4)      # 加入参数正则化   参数衰减
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(reset_optimizer, T_max=num_epochs)
    trian_model_losses = []     # model loss
    train_accuracies = []       # training accuracy
    test_accuracies  = []       # test accuracy
    spend_times = []            # Save training time per epoch
    train_contra_losses = []    # Distillation/contrastive loss
    test_top5_accuracies  = []  # Compute Top-5 accuracy

    featueAlign.to(device)
    reset_optimizer.add_param_group({'params': featueAlign.parameters()})

    for epoch_index in range(num_epochs):
        start_t4 = time.time()

        model.train()
        total_resnet_loss, correct = 0, 0    # Categorical cross-entropy loss
        total_contra_loss = 0                # Distillation/contrastive loss 
        for orig_input, label in tqdm(dataloader, desc=f"Epoch {epoch_index+1}", unit="batch"):
            
            orig_input, label = orig_input.to(device), label.to(device)
            
            # Whether to use carefully processed text descriptions
            if not labels_desc:
                text = [f"a photo of the {classes_name[i]}"for i in label]
            else:
                text = [labels_desc[classes_name[i]] for i in label]
            clip_inputs = clip_processor(text=text, images=orig_input, return_tensors="pt", padding=True, do_rescale=False)
            #  Move each tensor in clip_inputs to GPU device
            clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
            clip_outputs = clip_model(**clip_inputs)
            clip_img_embed = clip_outputs.image_embeds      # Extract image embeddings
            clip_text_embed = clip_outputs.image_embeds     # Generate text embeddings

            features, reset_resnet18_outputs = model(orig_input)
            features = featueAlign(features)
            contra_loss = contrastive_loss(features, clip_img_embed) + contrastive_loss(features, clip_text_embed)

            # Compute total loss
            reset_loss = crition(reset_resnet18_outputs, label)
            loss = 0.1*contra_loss + reset_loss
            print(f"Epoch {epoch_index+1} loss: ",  loss.item())
            reset_optimizer.zero_grad()   # Zero out gradients
            loss.backward()               # Update model parameters
            reset_optimizer.step()        # Apply gradient updates

            total_contra_loss += 0.1*contra_loss
            total_resnet_loss += loss.item()
            correct += (reset_resnet18_outputs.argmax(1) == label).sum().item()

            if epoch_index%20 == 0:
                # Create directory if not exists
                os.makedirs(os.path.join(save_path, name, "save"), exist_ok=True)
                torch.save(model.state_dict(), "{}/{}/save/{}_(CLIP){}_{}.pth".format(save_path, name, group_name, name, epoch_index))
                torch.save(featueAlign.state_dict(), "{}/{}/save/{}_(featueAlign){}_{}.pth".format(save_path, name, group_name, name, epoch_index))
                torch.save(reset_optimizer.state_dict(), "{}/{}/save/{}_(optimizer){}_{}.pth".format(save_path, name, group_name, name, epoch_index))
                torch.save(scheduler.state_dict(), "{}/{}/save/{}_(scheduler){}_{}.pth".format(save_path, name, group_name, name, epoch_index))
        
        scheduler.step()   # update learning rate

        trian_resnet_loss, train_accuracy =  total_resnet_loss/len(dataloader), correct/len(dataloader.dataset)
        total_contra_loss = total_contra_loss/len(dataloader)
        train_contra_losses.append(total_contra_loss)
        trian_model_losses.append(trian_resnet_loss)        
        train_accuracies.append(train_accuracy)  
        test_accuracy, top5_accuracy = evaluate_model_base(model, test_dataloader)
        test_accuracies.append(test_accuracy)
        test_top5_accuracies.append(top5_accuracy)
        print(f"Epoch{epoch_index+1}/{num_epochs}, Resnet_Loss: {trian_resnet_loss:.4f},",
                    f"Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
        print("Time elapsed for one epoch: ", (time.time()-start_t4)/60)
        spend_times.append((time.time()-start_t4)/60)
    # Initialize save directory
    os.makedirs(os.path.join(save_path, name, "img"), exist_ok=True)
    os.makedirs(os.path.join(save_path, name, "txt"), exist_ok=True)
    plot_img("Epoch", "Loss", "Training Loss Curve", "{}/{}/img/{}_training_loss_curve.png".format(save_path, name, group_name),
                                            [trian_model_losses, "Train Total Loss", "red"])
    plot_img("Epoch", "Accuracies", "Accuracies Curve", "{}/{}/img/{}_training_and_test_accuracies_curve.png".format(save_path, name, group_name), 
                                    [train_accuracies, "Train Accuracies", "red"], [test_accuracies, "Test Accuracies", "yellow"])
  
    # Log training metrics
    save_txt(trian_model_losses, "{}/{}/txt/{}__trian_losses.txt".format(save_path, name, group_name))
    save_txt(train_accuracies, "{}/{}/txt/{}_train_accuracies.txt".format(save_path, name, group_name))
    save_txt(test_accuracies, "{}/{}/txt/{}__test_accuracies.txt".format(save_path, name, group_name))
    save_txt(spend_times, "{}/{}/txt/{}__time_mins.txt".format(save_path, name, group_name))
    save_txt(train_contra_losses, "{}/{}/txt/{}__trian_contra_losses.txt".format(save_path, name, group_name))
    save_txt(test_top5_accuracies, "{}/{}/txt/{}__test_top5_accuracies.txt".format(save_path, name, group_name))


#  Train CLIP / classifier model with accuracy choose
def train_model_img2clip(num_epochs, clip_model, clip_processor, model, featueAlign, dataloader, classes_name, test_dataloader=None, save_path="./log", 
                          name="clip_model", group_name='base', labels_desc=None,):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crition = nn.CrossEntropyLoss() 
    reset_optimizer = optim.SGD(model.parameters(), lr=0.1,  momentum=0.9, weight_decay=1e-4)      # 加入参数正则化   参数衰减
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(reset_optimizer, T_max=num_epochs)
    trian_model_losses = []       # model loss  
    train_accuracies = []         # training accuracy  
    test_accuracies  = []         # test accuracy  
    spend_times = []              # Save training time per epoch  
    train_contra_losses = []      # Distillation/contrastive loss 
    test_top5_accuracies  = []    # Compute Top-5 accuracy  
  
    featueAlign.to(device)
    reset_optimizer.add_param_group({'params': featueAlign.parameters()})
    # reset_optimizer.load_state_dict(torch.load("./log/(CLIP)resnet-18(93)/save/1-(CLIP)resnet18-classifier_(optimizer)(CLIP)resnet-18_4.pth"))
    # scheduler.load_state_dict(torch.load("./log/(CLIP)resnet-18(93)/save/1-(CLIP)resnet18-classifier_(scheduler)(CLIP)resnet-18_4.pth"))


    for epoch_index in range(num_epochs):
        start_t4 = time.time()

        model.train()
        total_resnet_loss, correct = 0, 0    # Categorical cross-entropy loss
        total_contra_loss = 0                # Distillation/contrastive loss 
        for orig_input, label in tqdm(dataloader, desc=f"Epoch {epoch_index+1}", unit="batch"):
            
            orig_input, label = orig_input.to(device), label.to(device)
            
            if epoch_index > 1 and test_accuracies[epoch_index-1] < test_accuracies[epoch_index-2]:
                # Whether to use carefully processed text descriptions
                if not labels_desc:
                    text = [f"a photo of the {classes_name[i]}"for i in label]
                else:
                    text = [labels_desc[classes_name[i]] for i in label]
                clip_inputs = clip_processor(text=text, images=orig_input, return_tensors="pt", padding=True, do_rescale=False)
                # Move each tensor in clip_inputs to GPU device
                clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
                clip_outputs = clip_model(**clip_inputs)
                clip_img_embed = clip_outputs.image_embeds      # Extract image embeddings
                clip_text_embed = clip_outputs.image_embeds     # Generate text embeddings

                features, reset_resnet18_outputs = model(orig_input)
                features = featueAlign(features)
                contra_loss = contrastive_loss(features, clip_img_embed) + 0.5*contrastive_loss(features, clip_text_embed)
                # contra_loss = contrastive_loss(features, clip_text_embed)
                # contra_loss = contrastive_loss(features, clip_img_embed)

            else:
                contra_loss = 0
                _, reset_resnet18_outputs = model(orig_input)

            reset_loss = crition(reset_resnet18_outputs, label)
            # loss = 0.05*contra_loss + reset_loss
            # loss = 0.1*contra_loss + reset_loss
            # loss = 0.2*contra_loss + reset_loss
            # loss = 0.5*contra_loss + reset_loss
            loss = 0.8*contra_loss + reset_loss
            print(f"Epoch {epoch_index+1} loss: ",  loss.item())
            reset_optimizer.zero_grad()   # Zero out gradients
            loss.backward()               # Update model parameters
            reset_optimizer.step()        # Apply gradient updates

            total_contra_loss += 0.1*contra_loss
            total_resnet_loss += loss.item()
            correct += (reset_resnet18_outputs.argmax(1) == label).sum().item()

            if epoch_index%20 == 0:
                # Create directory if not exists
                os.makedirs(os.path.join(save_path, name, "save"), exist_ok=True)
                torch.save(model.state_dict(), "{}/{}/save/{}_(CLIP){}_{}.pth".format(save_path, name, group_name, name, epoch_index))
                torch.save(featueAlign.state_dict(), "{}/{}/save/{}_(featueAlign){}_{}.pth".format(save_path, name, group_name, name, epoch_index))
                torch.save(reset_optimizer.state_dict(), "{}/{}/save/{}_(optimizer){}_{}.pth".format(save_path, name, group_name, name, epoch_index))
                torch.save(scheduler.state_dict(), "{}/{}/save/{}_(scheduler){}_{}.pth".format(save_path, name, group_name, name, epoch_index))
        
        scheduler.step()   # update learning rate

        trian_resnet_loss, train_accuracy =  total_resnet_loss/len(dataloader), correct/len(dataloader.dataset)
        total_contra_loss = total_contra_loss/len(dataloader)
        train_contra_losses.append(total_contra_loss)
        trian_model_losses.append(trian_resnet_loss)        
        train_accuracies.append(train_accuracy)  
        test_accuracy, top5_accuracy = evaluate_model_base(model, test_dataloader)
        test_accuracies.append(test_accuracy)
        test_top5_accuracies.append(top5_accuracy)
        print(f"Epoch{epoch_index+1}/{num_epochs}, Resnet_Loss: {trian_resnet_loss:.4f},",
                    f"Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
        print("Time elapsed for one epoch: ", (time.time()-start_t4)/60)
        spend_times.append((time.time()-start_t4)/60)
    # Initialize save directory
    os.makedirs(os.path.join(save_path, name, "img"), exist_ok=True)
    os.makedirs(os.path.join(save_path, name, "txt"), exist_ok=True)
    plot_img("Epoch", "Loss", "Training Loss Curve", "{}/{}/img/{}_training_loss_curve.png".format(save_path, name, group_name),
                                            [trian_model_losses, "Train Total Loss", "red"])
    plot_img("Epoch", "Accuracies", "Accuracies Curve", "{}/{}/img/{}_training_and_test_accuracies_curve.png".format(save_path, name, group_name), 
                                    [train_accuracies, "Train Accuracies", "red"], [test_accuracies, "Test Accuracies", "yellow"])
  
    # Log training metrics
    save_txt(trian_model_losses, "{}/{}/txt/{}__trian_losses.txt".format(save_path, name, group_name))
    save_txt(train_accuracies, "{}/{}/txt/{}_train_accuracies.txt".format(save_path, name, group_name))
    save_txt(test_accuracies, "{}/{}/txt/{}__test_accuracies.txt".format(save_path, name, group_name))
    save_txt(spend_times, "{}/{}/txt/{}__time_mins.txt".format(save_path, name, group_name))
    save_txt(train_contra_losses, "{}/{}/txt/{}__trian_contra_losses.txt".format(save_path, name, group_name))
    save_txt(test_top5_accuracies, "{}/{}/txt/{}__test_top5_accuracies.txt".format(save_path, name, group_name))
