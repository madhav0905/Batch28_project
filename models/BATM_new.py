#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   BATM.py
@Time    :   2020/10/11 20:41:22
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''


from torch.nn.functional import normalize
from GPUtil import showUtilization as gpu_usage
##gpu_usage()  
import os
import re
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .gan import Generator, Encoder, Discriminator
import sys
sys.path.append('..')
from utils import evaluate_topic_quality, smooth_curve

# BATM model
class BATM:
    def __init__(self, bow_dim=1000, n_topic=20, hid_dim=512, device=None, taskname=None):
        self.n_topic = n_topic 
        self.bow_dim = bow_dim
        self.device = device
        self.id2token = None
        self.taskname = taskname
        
      
	
        self.generator = Generator(n_topic=n_topic,hid_dim=hid_dim,bow_dim=bow_dim)
        self.encoder = Encoder(bow_dim=bow_dim,hid_dim=hid_dim,n_topic=n_topic)
        self.discriminator = Discriminator(bow_dim=bow_dim,n_topic=n_topic,hid_dim=hid_dim)

        if device!=None:
            self.generator = self.generator.to(device)
            self.encoder = self.encoder.to(device)
            self.discriminator = self.discriminator.to(device)
            

    def train(self,train_data,batch_size=512, learning_rate=1e-4,test_data=None,num_epochs=1,is_evaluate=False,log_every=10,beta1=0,beta2=0.999,clip=0.01,n_critic=5):
    	 
        self.generator.eval()
        self.encoder.eval()
        self.discriminator.eval()
        self.id2token = {v:k for k,v in train_data.dictionary.token2id.items()}
        data_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True,num_workers=4, collate_fn=train_data.collate_fn,drop_last=True)
        print("inside train")
        optim_G = torch.optim.Adam(self.generator.parameters(),lr=learning_rate,betas=(beta1,beta2))
        optim_E = torch.optim.Adam(self.encoder.parameters(),lr=learning_rate,betas=(beta1,beta2))
        optim_D = torch.optim.Adam(self.discriminator.parameters(),lr=learning_rate,betas=(beta1,beta2))
        Gloss_lst, Eloss_lst, Dloss_lst = [], [], []
        #random_x=torch.rand(1,self.bow_dim,device='cuda')
        #poisson_dist=torch.poisson(random_x)
        """mean = 0.5
        stddev = 0.1
        sample = torch.randn(self.bow_dim,device='cuda')
        poisson_dist = torch.sigmoid(sample * stddev + mean)
        """
        mean = 0.5
        stddev = 0.25
        tensor = torch.normal(mean=mean, std=stddev, size=(1, self.bow_dim),device='cuda')
        tensor = torch.abs(tensor)
        poisson_dist=torch.sigmoid(tensor)
        print(poisson_dist.shape)	
        #print(self.generator.parameters())
        c_v_lst, c_w2v_lst, c_uci_lst, c_npmi_lst, mimno_tc_lst, td_lst = [], [], [], [], [], []
        for epoch in range(num_epochs):
            print("epoch is:",epoch)
            epochloss_lst = []
            for iter, data in enumerate(data_loader):
               
                
                txts, bows_real = data
                #print("iteration is ",bows_real.shape);
                bows_real = bows_real.to(self.device)
                bows_real /= torch.sum(bows_real,dim=1,keepdim=True)
                #self.bow_dim=self.bow_dim.int().to(self.device)
               	
                # Train Discriminator
                optim_D.zero_grad()
                
                #poisson_dist=torch.normal(mean=0.50,std=0.25,size=bows_real.shape,device='cuda')
                #poisson_dist=normalize(poisson_dis,p=2.0)
                #theta_fake = torch.from_numpy(np.random.dirichlet(alpha=1.0*np.ones(self.n_topic)/self.n_topic,size=(len(bows_real)))).float().to(self.device)
                #print(theta_fake.shape)
                theta_fake2=torch.from_numpy(np.random.laplace(loc=1.5,size=(len(bows_real) ,self.n_topic))).float().to(self.device)
                
                #print(theta_fake2)
                #print("using laplace")
                #print(theta_fake2.shape)
                #print("real vocab")
                #print(bows_real)
                #print("fake vocab")
                #print(poisson_dist)
                loss_D = -1.0*torch.mean(self.discriminator(self.encoder(bows_real).detach())) + torch.mean(self.discriminator(self.generator(theta_fake2,poisson_dist,batch_size)[0].detach()))

                loss_D.backward()
                optim_D.step()

                for param in self.discriminator.parameters():
                    param.data.clamp_(-clip,clip)
                
                if iter % n_critic==0:
                    # Train Generator
                   
     
                    l1_lambda=0.5
                    l1_norm=sum(p.abs().sum() for p in self.generator.parameters())
                    
                    optim_G.zero_grad()
                   
                    loss_G = -1.0*torch.mean(self.discriminator(self.generator(theta_fake2,poisson_dist,batch_size)[0]))+(l1_lambda*l1_norm)
                    #loss_G = -1.0*torch.mean(self.discriminator(self.generator(theta_fake2,poisson_dist,batch_size)))
                    
                    loss_G.backward()
                    
                    gpu_usage()
                    optim_G.step()
                    #print("after back")
                    #print(poisson_dist)
                    #print(theta_fake2)
                    # Train Encoder
                    optim_E.zero_grad()

                    loss_E = torch.mean(self.discriminator(self.encoder(bows_real)))

                    loss_E.backward()
                    optim_E.step()

                    Dloss_lst.append(loss_D.item())
                    Gloss_lst.append(loss_G.item())
                    Eloss_lst.append(loss_E.item())
                    print(f'Epoch {(epoch+1):>3d}\tIter {(iter+1):>4d}\tLoss_D:{loss_D.item():<.7f}\tLoss_G:{loss_G.item():<.7f}\tloss_E:{loss_E.item():<.7f}')
            
            if (epoch+1) % log_every == 0:
                print(f'Epoch {(epoch+1):>3d}\tLoss_D_avg:{sum(Dloss_lst)/len(Dloss_lst):<.7f}\tLoss_G_avg:{sum(Gloss_lst)/len(Gloss_lst):<.7f}\tloss_E_avg:{sum(Eloss_lst)/len(Eloss_lst):<.7f}')
                print('\n'.join([str(lst) for lst in self.show_topic_words()]))
                print('='*30)
                smth_pts_d = smooth_curve(Dloss_lst)
                smth_pts_g = smooth_curve(Gloss_lst)
                smth_pts_e = smooth_curve(Eloss_lst)
                plt.cla()
                plt.plot(np.array(range(len(smth_pts_g)))*log_every, smth_pts_g, label='loss_G')
                plt.plot(np.array(range(len(smth_pts_d)))*log_every, smth_pts_d, label='loss_D')
                plt.plot(np.array(range(len(smth_pts_e)))*log_every, smth_pts_e, label='loss_E')
                plt.legend()
                plt.xlabel('epochs')
                plt.title('Train Loss')
                plt.savefig('batm_trainloss.png')
                if test_data!=None:
                    c_v,c_w2v,c_uci,c_npmi,mimno_tc, td = self.evaluate(test_data,calc4each=False)
                    c_v_lst.append(c_v), c_w2v_lst.append(c_w2v), c_uci_lst.append(c_uci),c_npmi_lst.append(c_npmi), mimno_tc_lst.append(mimno_tc), td_lst.append(td)
        
    def evaluate(self, test_data, calc4each=False):
        topic_words = self.show_topic_words()
        return evaluate_topic_quality(topic_words, test_data, taskname=self.taskname,calc4each=calc4each)

    def show_topic_words(self, topic_id=None, topK=5):
        with torch.no_grad():
            topic_words = []
            
            idxes = torch.eye(self.n_topic).to(self.device)
            word_dist = self.generator.inference(idxes)
            vals, indices = torch.topk(word_dist, topK, dim=1)
            vals = vals.cpu().tolist()
            indices = indices.cpu().tolist()
            if topic_id == None:
                for i in range(self.n_topic):
                    topic_words.append([self.id2token[idx] for idx in indices[i]])
            else:
                topic_words.append([self.id2token[idx] for idx in indices[topic_id]])
            return topic_words
