#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   gan.py
@Time    :   2020/10/11 23:10:47
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat,bias=False)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    layers.append(nn.LeakyReLU(0.1, inplace=True))
    return layers

class Generator(nn.Module):
    def __init__(self,bow_dim,hid_dim,n_topic):
        super(Generator,self).__init__()
        #print(bow_dim)
        self.l1 = nn.L1Loss()
        
        self.f1=nn.Sequential(
            *block(bow_dim,n_topic),
            nn.Dropout(p=0.5),
            nn.Linear(n_topic,n_topic),
            nn.Softmax(dim=1)
        )
        self.g = nn.Sequential(
            *block(n_topic,n_topic),
            #*block(hid_dim,bow_dim,normalize=False),
            nn.Linear(n_topic,n_topic),
            #nn.L1norm()
            #need to apply l1 norma
            nn.Softmax(dim=1)
        )

    def inference(self,theta):
        return self.g(theta)
    
    def forward(self,theta,bow_di,bs):
        #print(self)
       
        x=bow_di.clone() #shape of 1 13290
        bow_dix=x.repeat(bs,1) 
        
       
	
        upper_layer=self.f1(bow_di)
        #print(upper_layer.shape)
        #print(" upper layer shape")
        bow_f = self.g(theta)
        #prinprint(bow_f.shape)
  
        word_count=torch.mul(bow_f,upper_layer)
        #print("the shape is ")
        #print(word_count.shape)
        #print(bow_dix.shape)        
        doc_f = torch.cat([bow_dix,word_count],dim=1)
        #print(doc_f.shape)
        loss = self.l1(theta, torch.zeros_like(theta))
        return doc_f,loss
        
class Generator2(nn.Module):
    def __init__(self,bow_dim,hid_dim,n_topic):
        super(Generator,self).__init__()
        self.g = nn.Sequential(
            *block(n_topic,hid_dim),
            #*block(hid_dim,bow_dim,normalize=False),
            nn.Linear(hid_dim,bow_dim),
            nn.Softmax(dim=1)
        )

    def inference(self,theta):
        return self.g(theta)
    
    def forward(self,theta):
        bow_f = self.g(theta)
        
        
        doc_f = torch.cat([theta,bow_f],dim=1)
       
        return doc_f

class Encoder(nn.Module):
    def __init__(self,bow_dim,hid_dim,n_topic):
        super(Encoder,self).__init__()
        self.e = nn.Sequential(
            *block(bow_dim,hid_dim),
            #*block(hid_dim,n_topic,normalize=False),
            nn.Linear(hid_dim,n_topic,bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self,bow):
        theta = self.e(bow)
        #print("theta_r")
        #print(theta)
        doc_r = torch.cat([theta,bow],dim=1)
       
        return doc_r

class Discriminator(nn.Module):
    def __init__(self,bow_dim,hid_dim,n_topic):
        super(Discriminator,self).__init__()
        self.d = nn.Sequential(
            *block(n_topic+bow_dim,hid_dim),
            #*block(hid_dim,1,normalize=False)
            nn.Linear(hid_dim,1,bias=True)
        )

    def forward(self,reps):
        # reps=[batch_size,n_topic+bow_dim]
        score = self.d(reps)
        return score
        
