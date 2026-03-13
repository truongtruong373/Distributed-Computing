import random

import torch
import torchvision
from collections import defaultdict
from tqdm import tqdm

import torchvision.transforms as transforms

def CIFAR10(batch_size=None, distribution=None, train = True):
    if train:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform= transform_train)

        label_to_indices = defaultdict(list)
        for idx, (_, label) in tqdm(enumerate(train_set)):
            label_to_indices[int(label)].append(idx)

        selected_indices = []
        for label, count in enumerate(distribution):
            selected_indices.extend(random.sample(label_to_indices[label], count))
        subset = torch.utils.data.Subset(train_set, selected_indices)

        train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

        return train_loader
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)
        return test_loader
