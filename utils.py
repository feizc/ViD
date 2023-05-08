import torch 
import numpy as np 
from PIL import Image 
import PIL 
from torchvision import transforms


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
    