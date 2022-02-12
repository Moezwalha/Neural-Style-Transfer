
##################################################
!pip install torch torchvision
!git clone https://github.com/Moezwalha/Neural-Style-Transfer
  
##################################################
import torch
from torchvision import models
from typing import Sized
from PIL import Image
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
##################################################

#Loading VGG Pretrained Model and remove the classifier part
vgg=models.vgg19(pretrained=True)
vgg= vgg.features 

#freeze the layers of the vgg model so any computation won't appear
for parameter in vgg.parameters():
  parameter.requires_grad_(False)

# move our model to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)
#################################################

def preprocess(img_path,max_size=500):
    image=Image.open(img_path).convert("RGB")
    if max(image.size) >  max_size:
      size= max_size   
    else:
      size=max(image.size)

    img_transform=T.Compose([
                             T.Resize(size),
                             T.ToTensor(), #convert image to tensor
                             T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    image=img_transform(image)
    image=image.unsqueeze(0) 

    return image

def deprocess(tensor):
  image=tensor.to("cpu").clone()
  image=torch.squeeze(image)  # (1,3,224,224)-->(3,224,224)
  image=image.numpy()
  image=image.transpose(1,2,0)
  image=image*np.array([0.229,0.224,0.225])+np.array([0.485,0.456,0.406])
  
  return image

def get_features(image,model): 
  layers={
      '0':'conv1_1',  #style_feature
      '5':'conv2_1',  #style_feature
      '10':'conv3_1', #style_feature
      '19':'conv4_1', #style_feature
      '21':'conv4_2', #content_feature
      '28':'conv5_1'  #style_feature
  }
  x=image
  features={}
  for name,layer in model._modules.items():
    x=layer(x)
    if name in layers:
      features[layers[name]]=x # x use the 0 layer and get the output=feature
      
  return features

def gram_matrix(tensor): # tensor=feature
  b,c,h,w=tensor.size()
  tensor=tensor.view(c,h*w) # change the tensor dim from (c,h,w) to (c,h*w)
  gram=torch.mm(tensor,tensor.t()) #mm =matrix multiplication
  
  return gram  # gram.size=(c,c)


def content_loss(target_conv4_2,content_conv4_2):
  loss=torch.mean((target_conv4_2-content_conv4_2)**2)
  return loss


style_weights={
    'conv1_1':1.0,
    'conv2_1':0.75,
    'conv3_1':0.2,
    'conv4_1':0.2,
    'conv5_1':0.2
}

def style_loss(style_weights,target_features,style_grams):
  loss=0
  for layer in style_weights:
    target_f=target_features[layer]
    target_gram=gram_matrix(target_f) 
    style_gram=style_grams[layer]
    b,c,h,w= target_f.shape
    layer_loss=style_weights[layer]* torch.mean((target_gram-style_gram)**2)
    loss+=layer_loss/(c*h*w)

  return loss

def total_loss(c_loss,s_loss,alpha,beta):
  loss=alpha*c_loss+beta*s_loss
  return loss

def main():
  
  # preprocess content and style images
  content_img_preprocessed = preprocess('/content/Neural-Style-Transfer/images/content_image/city.jpg').to(device)
  style_img_preprocessed   = preprocess('/content/Neural-Style-Transfer/images/style_image/sky.jpg').to(device)
  
  # extract features from content and style images
  content_features=get_features(content_img_preprocessed,vgg)
  style_features=get_features(style_img_preprocessed,vgg)
  
  # calculate the  gram matrix of every feature
  style_grams={layer: gram_matrix(style_features[layer]) for layer in style_features}

  #select the target image
  target_image=content_img_preprocessed.clone().requires_grad_(True).to(device)
  # hyperparameters(personal choice)
  optimizer=optim.Adam([target_image] , lr=0.003)
  alpha=1
  beta=1e5
  epochs=3000
  show_every = 500
  results=[]
  
  
  # calculate the total loss in each epoch
  for i in range(epochs):
    target_f=get_features(target_image,vgg)
    #calculate the content loss
    c_loss=content_loss(target_f['conv4_2'],content_features['conv4_2'])
    #calculate the style loss 
    s_loss=style_loss(style_weights, target_f,style_grams)
    #calculate the total loss
    t_loss=total_loss(c_loss,s_loss,alpha,beta)
    #optimize our target image pixels
    optimizer.zero_grad()
    t_loss.backward()
    optimizer.step()

    if (i % show_every ==0):
      print("total loss of the {} = {}".format(i,t_loss))
      results.append(deprocess(target_image.detach()))
   
  return target_image,content_img_preprocessed
    
#######################################
main()

#######################################
#See the result
target_copy = deprocess(target_image.detach())
content_copy = deprocess(content_img_preprocessed)
fig,(ax1,ax2) = plt.subplots(1,2,figsize = (10,5))
ax1.imshow(target_copy)
ax2.imshow(content_copy)
