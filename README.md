## ViD: Text-to-Image Diffusion Models as Visual Learners

We use the feature map from text-to-image diffusion model with a linear projection as visual representation. 
Specifically, we select several feature map from U-Net when feed the image with scheme t = 0.  

We only train the linear network on the ImageNet-1k and evaluate the performance as:

| Model | Top-1 Accuracy |
  |------|------|
  | ViD | 60.2 | 


After search for literatures, we think that U-Net structure is not suitable for image classification, and more exploration for segmentation is deserved. 

