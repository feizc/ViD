import os 
import torch 
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler 
from transformers import CLIPTextModel, CLIPTokenizer 


class ViDModel(): 
    def __init__(
        self,
        sd_model_path,
        device,
        index_list = [4, 6, 8],
        kernel_list = [(8, 8), (32, 32), (64, 64)],
    ):

        tokenizer = CLIPTokenizer.from_pretrained(
            os.path.join(sd_model_path, 'tokenizer')
        )
        text_encoder = CLIPTextModel.from_pretrained(
            os.path.join(sd_model_path, "text_encoder")
        )
        vae = AutoencoderKL.from_pretrained(
            os.path.join(sd_model_path, "vae")
        )
        unet = UNet2DConditionModel.from_pretrained(
            os.path.join(sd_model_path, "unet")
        )
        self.noise_scheduler = DDPMScheduler.from_config(sd_model_path, subfolder="scheduler") 
        vae = vae.to(device)
        unet = unet.to(device) 
        text_encoder = text_encoder.to(device) 

        self.vae = vae.eval()
        self.unet = unet.eval() 
        self.text_encoder = text_encoder.eval() 

        self.input_ids = tokenizer(
            " ",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        self.index = index_list 
        self.avg1 = torch.nn.AvgPool2d(kernel_list[0]) 
        self.avg2 = torch.nn.AvgPool2d(kernel_list[1]) 
        self.avg3 = torch.nn.AvgPool2d(kernel_list[2]) 
        self.projection = torch.nn.Linear(2880, 1000)

    def extract_feature(self, img,): 
        """Extract features from images.""" 
        with torch.no_grad(): 
            latents = self.vae.encode(img).latent_dist.sample().detach()
            latents = latents * 0.18215 

            noise = torch.randn(latents.shape).to(latents.device)
            bsz = latents.shape[0] 

            timesteps = torch.zeros((bsz,),device=latents.device).long() 
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = self.text_encoder(self.input_ids)[0]

            outputs = self.unet(noisy_latents, timesteps, encoder_hidden_states)

        selected_outputs = [] 
        for index in self.index_list: 
            selected_outputs.append(outputs[index]) 
        return selected_outputs 

    def forward(self, img): 
        selected_outputs = self.extract_feature(img,) 
        feature_map1 = self.avg1(selected_outputs[0]).squeeze(3).squeeze(2)
        feature_map2 = self.avg2(selected_outputs[1]).squeeze(3).squeeze(2)
        feature_map3 = self.avg3(selected_outputs[2]).squeeze(3).squeeze(2)
        
        feature_map = torch.cat((feature_map1, feature_map2, feature_map3), dim=1) 
        logits = self.projection(feature_map) 
        return logits 

    