import torch
import torchvision
import torchvision.transforms as transforms
from utils import DataIterator

def _fix_cls_to_idx(ds):
    for cls in ds.class_to_idx:
        ds.class_to_idx[cls] = int(cls)

def prepare_data(args):
    num_classes = 14

    # resnet recommended normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # transform
    # Note: rescaling to 224 and center-cropping already processed in img folders    
    transform = transforms.Compose([
        transforms.ToTensor(), # to [0,1]
        normalize
    ])
    
    train_data_gold = torchvision.datasets.ImageFolder('data/clothing1M/clean_train', transform=transform)
    train_data_silver = torchvision.datasets.ImageFolder('data/clothing1M/noisy_train', transform=transform)
    val_data = torchvision.datasets.ImageFolder('data/clothing1M/clean_val', transform=transform)
    test_data = torchvision.datasets.ImageFolder('data/clothing1M/clean_test', transform=transform)

    # fix class idx to equal to class name
    _fix_cls_to_idx(train_data_gold)
    _fix_cls_to_idx(train_data_silver)
    _fix_cls_to_idx(val_data)
    _fix_cls_to_idx(test_data)

    gold_sampler = None
    silver_sampler = None
    val_sampler = None
    test_sampler = None
    batch_size = args.bs
        
    train_gold_loader = DataIterator(torch.utils.data.DataLoader(train_data_gold,   batch_size=batch_size, shuffle=(gold_sampler is None),
                                                                 num_workers=args.prefetch, pin_memory=True, sampler=gold_sampler))
    train_silver_loader = torch.utils.data.DataLoader(train_data_silver, batch_size=batch_size, shuffle=(silver_sampler is None),
                                                      num_workers=args.prefetch, pin_memory=True, sampler=silver_sampler)
    val_loader  = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=(val_sampler is None),
                                              num_workers=args.prefetch, pin_memory=True, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=(test_sampler is None),
                                              num_workers=args.prefetch, pin_memory=True, sampler=test_sampler)

    return train_gold_loader, train_silver_loader, val_loader, test_loader, num_classes
