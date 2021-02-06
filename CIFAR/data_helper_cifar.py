import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from utils import DataIterator

def prepare_data(gold_fraction, corruption_prob, corruption_type, args):
    if args.use_mwnet_loader:
        return prepare_data_mwnet(gold_fraction, corruption_prob, corruption_type, args)
    else:
        return prepare_data_mlc(gold_fraction, corruption_prob, corruption_type, args)

def prepare_data_mwnet(gold_fraction, corruption_prob, corruption_type, args):
    from load_corrupted_data_mlg import CIFAR10, CIFAR100    
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if True: # no augment as used by mwnet
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    args.num_meta = int(50000 * gold_fraction)

    if args.dataset == 'cifar10':
        num_classes = 10
        
        train_data_meta = CIFAR10(
            root=args.data_path, train=True, meta=True, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR10(
            root=args.data_path, train=True, meta=False, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR10(root=args.data_path, train=False, transform=test_transform, download=True)

        valid_data = CIFAR10(
            root=args.data_path, train=True, meta=True, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)

    elif args.dataset == 'cifar100':
        num_classes = 100
        
        train_data_meta = CIFAR100(
            root=args.data_path, train=True, meta=True, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR100(
            root=args.data_path, train=True, meta=False, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR100(root=args.data_path, train=False, transform=test_transform, download=True)

        valid_data = CIFAR100(
            root=args.data_path, train=True, meta=True, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)        

    train_gold_loader = DataIterator(torch.utils.data.DataLoader(train_data_meta, batch_size=args.bs, shuffle=True,
        num_workers=args.prefetch, pin_memory=True))
    train_silver_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.bs, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    return train_gold_loader, train_silver_loader, valid_loader, test_loader, num_classes

def prepare_data_mlc(gold_fraction, corruption_prob, corruption_type, args):
    from load_corrupted_data import CIFAR10, CIFAR100
        
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    # since cifar10 and cifar100 have no official validation split, use gold as valid also
    if args.dataset == 'cifar10':
        train_data_gold = CIFAR10(
            args.data_path, True, True, gold_fraction, corruption_prob, args.corruption_type,
            transform=train_transform, download=True, distinguish_gold=False, seed=args.seed)
        train_data_silver = CIFAR10(
            args.data_path, True, False, gold_fraction, corruption_prob, args.corruption_type,
            transform=train_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices, seed=args.seed, distinguish_gold=False, weaklabel=args.weaklabel) # note here for the change
        train_data_gold_deterministic = CIFAR10(
            args.data_path, True, True, gold_fraction, corruption_prob, args.corruption_type,
            transform=test_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices, distinguish_gold=False, seed=args.seed)
        test_data = CIFAR10(args.data_path, train=False, transform=test_transform, download=True, distinguish_gold=False, seed=args.seed)

        # same as gold
        valid_data = CIFAR10(
            args.data_path, True, True, gold_fraction, corruption_prob, args.corruption_type,
            transform=train_transform, download=True, distinguish_gold=False, seed=args.seed)

        num_classes = 10

    elif args.dataset == 'cifar100':
        train_data_gold = CIFAR100(
            args.data_path, True, True, gold_fraction, corruption_prob, args.corruption_type,
            transform=train_transform, download=True, distinguish_gold=False, seed=args.seed)
        train_data_silver = CIFAR100(
            args.data_path, True, False, gold_fraction, corruption_prob, args.corruption_type,
            transform=train_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices, seed=args.seed, distinguish_gold=False,
            weaklabel=args.weaklabel) # note the weaklabel arg
        train_data_gold_deterministic = CIFAR100(
            args.data_path, True, True, gold_fraction, corruption_prob, args.corruption_type,
            transform=test_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices, distinguish_gold=False, seed=args.seed)
        test_data = CIFAR100(args.data_path, train=False, transform=test_transform, download=True, distinguish_gold=False, seed=args.seed)

        # same as gold
        valid_data = CIFAR100(
            args.data_path, True, True, gold_fraction, corruption_prob, args.corruption_type,
            transform=train_transform, download=True, distinguish_gold=False, seed=args.seed)
        
        num_classes = 100


    gold_sampler = None
    silver_sampler = None
    valid_sampler = None
    test_sampler = None
    batch_size = args.bs
        
    train_gold_loader = DataIterator(torch.utils.data.DataLoader(
        train_data_gold,   batch_size=batch_size, shuffle=(gold_sampler is None),
        num_workers=args.prefetch, pin_memory=True, sampler=gold_sampler))
    train_silver_loader =torch.utils.data.DataLoader(
        train_data_silver, batch_size=batch_size, shuffle=(silver_sampler is None),
        num_workers=args.prefetch, pin_memory=True, sampler=silver_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=(valid_sampler is None), num_workers=args.prefetch, pin_memory=True, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=(test_sampler is None), num_workers=args.prefetch, pin_memory=True, sampler=test_sampler)

    return train_gold_loader, train_silver_loader, valid_loader, test_loader, num_classes
