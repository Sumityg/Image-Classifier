import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets,transforms,models
from PIL import Image
import json
import sys

def agr_paser():
    paser=argparse.ArgumentParser(description='Predictor file')
    paser.add_argument('--checkpoint',type=str,default='checkpoint.pth',help='saving the model')
    paser.add_argument('--gpu',type=bool,default='True',help='True:gpu,False:cpu')
    paser.add_argument('--top_k',type=int,default=5,help='top classes')
    paser.add_argument('--image',type=str,required='True',help='Path where the image is stored')
    
    args=paser.parse_args()
    return args

def load_checkpoint(path):
    checkpoint=torch.load(path)
    model=getattr(models,checkpoint['structure'])(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad=False
        
    model.classifier=checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dic'])
    model.class_to_idx=checkpoint['class_to_idx']
    
    return model

def process_image(image):
    preprocess=transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    pil_image=Image.open(image)
    np_image=preprocess(pil_image).numpy()
    
    
    return np_image

def predict(image_path,model,device,cat_to_name,topk):
    model.eval()
    model.to(device)
    image=process_image(image_path)
    image=torch.tensor(image)
    image.unsqueeze_(0)
    logp=model(image.to(device))
    p=torch.exp(logp)
    probs,top_class=torch.topk(p,topk,sorted=True)
    
   
    #Converting the indices to class
    idx_to_class={val:key for key,val in model.class_to_idx.items()}
    prob_arr=probs.cpu().data.numpy()[0]
    pred_indexes=top_class.cpu().data.numpy()[0].tolist()
    pred_labels=[idx_to_class[x] for x in pred_indexes]
    pred_class=[cat_to_name[str(x)] for x in pred_labels]
    
    
    return pred_labels,pred_class,prob_arr

def main():
    args=agr_paser()
    
    with open('cat_to_name.json','r') as f:
        cat_to_name=json.load(f)
        
    is_gpu = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    if is_gpu and use_cuda:
        device = torch.device("cuda:0")
        print(f"Device is set to {device}")

    else:
        device = torch.device("cpu")
        print(f"Device is set to {device}")
        
    model = load_checkpoint(args.checkpoint)
    np_image = process_image(args.image)
    topk_label,topk_class,topk_probability = predict(args.image, model, device, cat_to_name, args.top_k)
    
    print('Predicted top classes : ', topk_class)
    print('Flowers: ', topk_label)
    print('Probablity: ', topk_probability)
    
if __name__=='__main__':main()
    

    
    