import torch 
import os 
import numpy as np 
from PIL import Image 
import PIL 
from torchvision import transforms, datasets 


def image_preprocess(img, size=512, center_crop=False, flip_p=0.5): 
    img = np.array(img).astype(np.uint8) 
    if center_crop:
        crop = min(img.shape[0], img.shape[1])
        h, w, = (
            img.shape[0],
            img.shape[1],
        )
        img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
    image = Image.fromarray(img)
    image = image.resize((size, size), resample=PIL.Image.BICUBIC)

    image = transforms.RandomHorizontalFlip(p=flip_p)(image)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    return torch.from_numpy(image).permute(2, 0, 1) 
    

def build_dataset(is_train, args): 
    root = os.path.join(args.data_path, 'train' if is_train else 'val') 
    dataset = datasets.ImageFolder(root, transform=image_preprocess) 
    print(dataset)
    return dataset 


import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr