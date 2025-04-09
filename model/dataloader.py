# Load dataset

import PIL.Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import PIL
import numpy


def get_dataset(is_train=True, dowanload=True, dataset="cifar100", root="./data" ,is_transform="train"):

    # Apply image transformations
    if is_transform == "train":
        transform_dataset = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Color Jittering
                                transforms.ToTensor(),
                                # transforms.Normalize([-0.2875, -0.2977, -0.3175], [0.3049, 0.2911, 0.2955])
                                # transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                            ])
    else:
        transform_dataset = transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize([-0.2875, -0.2977, -0.3175], [0.3049, 0.2911, 0.2955])
                                # transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                            ])
    
    assert dataset in ["cifar100", "cifar10"], "please choose the proper dataset!"     # Validate dataset availability 
    # Download dataset if missing
    if dataset == "cifar100":
        res_dataset = torchvision.datasets.CIFAR100(root=root, train=is_train, 
                                                      download=dowanload, transform=transform_dataset)
    # image, label = res_dataset[12]
    # image.save("82.jpg")
    return res_dataset


def get_dataset_clip(is_train=True, dowanload=True, dataset="cifar100", root="./data" ,is_transform="train"):

    # Perform Image Transformations
    if is_transform == "train":
        transform_dataset = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15),
                                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Color Jittering
                                transforms.ToTensor(),
                                # transforms.Normalize([-0.2875, -0.2977, -0.3175], [0.3049, 0.2911, 0.2955])
                                # transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                            ])
    else:
        transform_dataset = transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize([-0.2875, -0.2977, -0.3175], [0.3049, 0.2911, 0.2955])
                                # transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                            ])
    
    assert dataset in ["cifar100", "cifar10"], "please choose the proper dataset!"     # Validate dataset availability 
    # Download Dataset
    if dataset == "cifar100":
        res_dataset = torchvision.datasets.CIFAR100(root=root, train=is_train, 
                                                      download=dowanload, transform=transform_dataset)
    # image, label = res_dataset[12]
    # image.save("82.jpg")
    return res_dataset

def get_dataloader(dataset, shuffle=True, batch_size=512, numer_workers=16):
    res_loader = DataLoader(dataset, shuffle=shuffle, num_workers=numer_workers, 
                                      batch_size=batch_size)
    return dataset.classes, res_loader 


import os

# is_train=True, dowanload=True, dataset="cifar100", root="./data" ,is_transform="train"
# shuffle=True, batch_size=512, numer_workers=16

def load_data(dataset="cifar100", is_train=True, dowanload=True, root="./data", shuffle=True, batch_size=256, numer_workers=16):
    if dataset == "cifar100":
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            # transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(root=root, train=is_train, download=dowanload, transform=transform_train),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=numer_workers
        )

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100('data', train=False, transform=transform_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=numer_workers
        )

    elif dataset == "IMAGENET":
        # traindir = os.path.join(args.data, 'train')
        # valdir = os.path.join(args.data, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = torchvision.datasets.ImageFolder(
            root,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        # Check class labels
        # print(train_dataset.classes)

        # if args.distributed:
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # else:
        #     train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            # shuffle=(train_sampler is None),
            shuffle=shuffle,
            num_workers=numer_workers,
            pin_memory=True,
            # sampler=train_sampler
        )

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(root, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size,
            shuffle=False,
            num_workers=numer_workers,
            pin_memory=True
        )

    return train_loader, test_loader


def get_imagenet_classes(root):
    traindir = os.path.join(root, 'train')
    dataset = torchvision.datasets.ImageFolder(traindir)
    # Build a mapping from class indices to class names.
    class_idx = dataset.class_to_idx
    idx_class = {v: k for k, v in class_idx.items()}  # Invert the dictionary to map indices to class labels.
    return idx_class


# Calculate mean and std of dataloader
def get_mean_std(mydataloader):

    # Accumulate sum and squared sum per channel
    total_sum = torch.zeros(3)
    total_squared_sum = torch.zeros(3)
    num_batches = 0
    
    for images, _ in mydataloader:
        # Reshape batch to [batch_size, channels, height*width]
        images = images.view(images.shape[0], images.shape[1], -1)  # Parameter -1 for auto-sizing

        # Compute per-channel mean and squared sum
        total_sum += images.mean(dim=[0, 2])
        total_squared_sum += images.pow(2).mean(dim=[0, 2])
        num_batches += 1
    
    # Calculate dataset-wide mean and standard deviation
    mean = total_sum / num_batches
    std = (total_squared_sum / num_batches - mean**2).sqrt()

    # Return mean and std
    return mean, std


if __name__ == "__main__":
    mean = torch.tensor([-0.2875, -0.2977, -0.3175])
    std = torch.tensor([0.3049, 0.2911, 0.2955])
    dataset = get_dataset()
    class_names, dataloader = get_dataloader(dataset)
    print(class_names[82])
    meean, std = get_mean_std(dataloader)
    print("mean and std: ", mean, std)
    for image, label in dataloader:
        print(image.shape, label)
        break