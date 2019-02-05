import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torchvision import datasets,transforms,models
import torch.nn.functional as F
from collections import OrderedDict
import json
from torch.autograd import Variable
import argparse
import os
import sys

def agr_paser():
    paser=argparse.ArgumentParser(description='trainer file')
    
    paser.add_argument('--data_dir',type=str,default='flowers',help='dataset directory')
    paser.add_argument('--gpu',type=bool,default='True',help='True:gpu,False:cpu')
    paser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    paser.add_argument('--epochs',type=int,default=10,help='number of epochs')
    paser.add_argument('--arch',type=str,default='vgg11',help='architecture')
    paser.add_argument('--hidden_units',type=int,default=[600,200],help='hidden units for layer')
    paser.add_argument('--save_dir',type=str,default='checkpoint.pth',help='save trained model to disk')
    
    args=paser.parse_args()
    return args

def process_data(train_dir,test_dir,valid_dir):
    train_transforms =transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224,                                                                       0.225))]) 
                                      
    test_transforms=transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224,                                                  0.225))])

    valid_transforms=transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224,                                                  0.225))])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data=datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data=datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    trainloader=torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    testloader=torch.utils.data.DataLoader(test_data,batch_size=64)
    validloader=torch.utils.data.DataLoader(valid_data,batch_size=64)
    
    return trainloader,testloader,validloader,train_data,test_data,valid_data

def basic_model(arch):
    #Load the pretrained network
    if arch==None or arch=='vgg16':
        load_model=models.vgg16(pretrained=True)
        print('Use vgg16')
    else:
        print('Use vgg16 or densenet only defaulting to vgg16')
        load_model=models.vgg16(pretrained=True)
        
    return load_model

def set_classifier(model,hidden_units):
    if hidden_units==None:
        hidden_units=500
    input=model.classifier[0].in_features
    classifier=nn.Sequential(OrderedDict([('fc1', nn.Linear(input,hidden_units,bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout',nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 128)),
                          ('relu2', nn.ReLU()),
                          ('dropout',nn.Dropout(p=0.5)),
                          ('fc3', nn.Linear(128, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    return model

def train_model(epochs,trainloader,validloader,gpu,model,optimizer,criterion):
    if type(epochs)==type(None):
        epochs=10
        print("Epochs=10")
    train_losses,validation_losses=[],[]
    model.to('cuda')

    for e in range(epochs):
        running_loss=0
        for images,labels in trainloader:
            images,labels = images.to('cuda'),labels.to('cuda')
            optimizer.zero_grad()
            logps=model.forward(images)
            loss=criterion(logps,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        
        else:
            validation_loss=0
            accuracy=0
        
        #Turning off the gradient for validation,saves memory and computations
            with torch.no_grad():
                for images,labels in validloader:
                    images,labels=images.to('cuda'),labels.to('cuda')
                    logps=model.forward(images)
                    batch_loss=criterion(logps,labels)
                    validation_loss+=batch_loss.item()
                
                    ps=torch.exp(logps)
                    top_p,top_class=ps.topk(1,dim=1)
                    equals=top_class==labels.view(*top_class.shape)
                    accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
        
        
            train_losses.append(running_loss/len(trainloader))
            validation_losses.append(validation_loss/len(validloader))
        
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Validation Loss: {:.3f}.. ".format(validation_loss/len(validloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

    return model

def valid_model(epochs,model,testloader,gpu,criterion):
    if type(epochs)==type(None):
        epochs=10
        print("Epochs=10")
    test_losses=[]
    for e in range(epochs):
        test_loss=0
        accuracy=0
        model.eval()
        with torch.no_grad():
            for images,labels in testloader:
                images,labels=images.to('cuda'),labels.to('cuda')
                logps=model.forward(images)
                batch_loss=criterion(logps,labels)
                test_loss+=batch_loss.item()
                ps=torch.exp(logps)
                top_p,top_class=ps.topk(1,dim=1)
                equals=top_class==labels.view(*top_class.shape)
                accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
        
                test_losses.append(test_loss/len(testloader))

    print("Epoch: {}/{}.. ".format(e+1, epochs),
           "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
           "Test Accuracy: {:.3f}".format(accuracy/len(testloader))) 
    
    
def save_checkpoint(model,train_data,save_dir,arch):
    model.class_to_idx=train_data.class_to_idx
    checkpoint={'structure':arch,
                'classifier':model.classifier,
                'state_dic':model.state_dict(),
                'class_to_idx':model.class_to_idx}
    return torch.save(checkpoint,save_dir)

def main():
    args=agr_paser()
    
    data_dir='flowers'
    train_dir=data_dir+'/train'
    valid_dir=data_dir+'/valid'
    test_dir=data_dir+'/test'
    trainloader,testloader,validloader,train_data,test_data,valid_data=process_data(train_dir,test_dir,valid_dir)
    model=basic_model(args.arch)
    
    for param in model.parameters():
        param.requires_grad=False
        
    model=set_classifier(model,args.hidden_units)
    
    criterion=nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(),lr=args.lr)
    trmodel=train_model(args.epochs,trainloader,validloader,args.gpu,model,optimizer,criterion)
    valid_model(args.epochs,trmodel,testloader,args.gpu,criterion)
    save_checkpoint(trmodel,train_data,args.save_dir,args.arch)
    print('Done')
    
if __name__=='__main__':main()
    
    
    