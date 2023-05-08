import os 
import torch 
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler 
from transformers import CLIPTextModel, CLIPTokenizer 
from PIL import Image 

from utils import image_preprocess


model_path = './ckpt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


tokenizer = CLIPTokenizer.from_pretrained(
    os.path.join(model_path, 'tokenizer')
)
text_encoder = CLIPTextModel.from_pretrained(
    os.path.join(model_path, "text_encoder")
)
vae = AutoencoderKL.from_pretrained(
    os.path.join(model_path, "vae")
)
unet = UNet2DConditionModel.from_pretrained(
    os.path.join(model_path, "unet")
)
noise_scheduler = DDPMScheduler.from_config(model_path, subfolder="scheduler") 

vae = vae.to(device) 
unet = unet.to(device) 

vae.eval()
unet.eval() 

img = Image.open('cat.png').convert("RGB") 
img = image_preprocess(img)
caption = ' '
input_ids = tokenizer(
    caption,
    padding="max_length",
    truncation=True,
    max_length=tokenizer.model_max_length,
    return_tensors="pt",
).input_ids[0]


latents = vae.encode(img.to(device)).latent_dist.sample().detach()
latents = latents * 0.18215 
noise = torch.randn(latents.shape).to(latents.device)
bsz = latents.shape[0] 
timesteps = torch.randint(0, 0, (bsz,), device=latents.device).long() 
noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
encoder_hidden_states = text_encoder(input_ids.to(device))[0]
noise_pred = unet(latents, timesteps, encoder_hidden_states).sample 

print(noise_pred)







