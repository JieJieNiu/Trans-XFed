#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:35:22 2024

@author: jie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:02:44 2023

@author: jolie
"""

import copy
import torch
from tqdm import tqdm
from model import TransFed
import numpy as np
from client import train, test, get_val_loss
from get_data import nn_seq_wind
import args
import random
import tenseal as ts
from datetime import datetime
from torch.nn import init
import torch.nn as nn
import pandas as pd
'''
nn：global model
nns：local models
'''

def initialize(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data)
            
class FedProx:
    def __init__(self, args):
        self.args = args
        self.nn = TransFed(args=self.args, name='server').to(args.device)
        self.nn.apply(initialize)
        self.total_params = sum(p.numel() for p in self.nn.parameters() if p.requires_grad)
        print ("TOTAL PARAMETERS",self.total_params)
        self.nns = []
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.args.clients[i]
            self.nns.append(temp)
        #Generate keys
        self.context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=8192,
                    coeff_mod_bit_sizes=[60, 40, 40, 60]
                  )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40 
        print('[{}] FHE publick keys and private keys generated...'.format(datetime.now()))
    
    def server(self):
        chosen_clients_list=[]
        convergence=[]
        for t in tqdm(range(self.args.r)):
            #communication rounds self.args.r #
            print('round', t + 1, ':')
            # sampling
            round=t+1
            clients_index=np.array(range(self.args.K))
            clients_list=list(clients_index)
            m = int(self.args.C * self.args.K)
            if self.args.fixmu == True:
                self.mu=t/(self.args.r*1000)
            else:
                self.mu=self.args.mu            
            # dispatch the global model to all clients, choose the higher loss of rate C.
            self.dispatch(clients_list)
            
            #Choose clients
            if self.args.poc == True:
                chosen_clients = self.power_of_choice(clients_list, clients_index, m)
                print(chosen_clients)
                chosen_clients_list.append(chosen_clients)
            elif self.args.poc == False:
                chosen_clients = random.sample(clients_list,2)
                print(chosen_clients)
                chosen_clients_list.append(chosen_clients)
            # local updating choosen clients paticipants the training
            self.client_update(chosen_clients)

            # aggregation the choosen clients model
            self.aggregation(chosen_clients)
            
            test_loss, recall_test, percision_test, F1_test=self.global_test('test')
            print('test_loss {} recall_test{} percision_test{} F1_test{}'.format(test_loss, recall_test, percision_test, F1_test))
            results=[test_loss, recall_test, percision_test, F1_test]
            convergence.extend(results)
            torch.save({'state_dict': self.nn.state_dict()},
                   args.MODEL_SAVE + args.MODEL_NAME +'_'+ format(round) + '_last.pkl')
        
        print(chosen_clients_list)
        print(convergence)
        #pd.DataFrame(chosen_clients_list).to_csv('clients.csv',args.LOG_PATH,header=True)
        #pd.DataFrame(convergence).to_csv('convergence.csv',args.LOG_PATH,header=True)
        return self.nn
    

    def aggregation(self, index):
        s=0
        total_sample=0
        for j in index:
            s += self.nns[j].len
            total_sample += self.args.samplesize[j]
            self.weights=[(1-self.args.samplesize[j]/total_sample),self.args.samplesize[j]/total_sample]
            
        params = {}
        for k, v in self.nns[0].named_parameters():
            params[k] = torch.zeros_like(v.data)

        #weighted average，the weight is the size of samples of each clients.
        
        parameters=[]
        parameters_list=[]
        def split_list(parameters,size):
            return [parameters[j:j+size] for j in range(0, len(parameters), size)]
        
        #FHE encryption
        
        for i in index:
            for k, v in self.nns[i].named_parameters():
                parameters += v.data.flatten(0).numpy().tolist()

                params[k] += v.data *(self.nns[i].len/s)
       
        parameters_list = split_list(parameters, self.total_params)
        print("Number of parameters collected after loop:", len(parameters_list))
        
        enc_params_1=ts.ckks_vector(self.context,parameters_list[0])
        enc_params_2=ts.ckks_vector(self.context,parameters_list[1])
        
        if self.args.wa == False:
            enc_params=(enc_params_1+enc_params_2)*0.5
        if self.args.wa == True: 
            print("aggragation weights", self.weights)
            enc_params=enc_params_1*self.weights[0]+enc_params_2*self.weights[1]
        
        #FHE decryption
        dec_params=enc_params.decrypt()
        
        #update the global model
        params = {}
        shapes=[]
        for k, v in self.nn.named_parameters():
            
            params[k] = torch.zeros_like(v.data)
            
            shapes.append(v.data.shape)
            
        reshaped_params = []
        start_idx = 0
        for shape in shapes:
            num_elements = torch.tensor(shape).prod().item()
            reshaped_param = torch.Tensor(dec_params)[start_idx:start_idx + num_elements].view(shape)
            reshaped_params.append(reshaped_param)
            start_idx += num_elements
            
        for v, reshaped_param in zip(self.nn.parameters(), reshaped_params):
            v.data=reshaped_param.clone()

        
    def dispatch(self, clients_list):
        #将全局模型的参数分发给选中的客户端模型，以确保每个客户端在开始训练之前都具有最新的全局模型参数
        for j in clients_list:
            client_loss=[]
            #代码使用 zip 函数同时迭代选中的客户端模型 self.nns[j] 的参数和全局模型 self.nn 的参数。这意味着对于每个客户端模型和全局模型都会迭代对应的参数。
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()
    
    def power_of_choice(self, clients_list, clients_index, m):
        client_f1=[]
        clients = ['data' + str(i) for i in range(0, 4)]
        for j in clients_list:
            model=self.nn
            model.eval()
            _,_,_,F1_test=test(self.args, model, clients[j])
            client_f1.append(F1_test)

        sort_id=np.array(client_f1).argsort().tolist()
        chosen_clients = list(np.array(clients_index)[sort_id][(self.args.K-m):])
        print("chossen clients", chosen_clients)
        return chosen_clients

    # def power_of_choice(self, clients_list, clients_index, m):
    #     client_loss=[]
    #     clients = ['data' + str(i) for i in range(0, 4)]
    #     for j in clients_list:
    #         model=self.nn
    #         model.eval()
    #         test_loss,_,_,F1_test=test(self.args, model, clients[j])
    #         client_loss.append(test_loss)

    #     sort_id=np.array(client_loss).argsort().tolist()
    #     chosen_clients = list(np.array(clients_index)[sort_id][(self.args.K-m):])
    #     print("chossen clients", chosen_clients)
    #     return chosen_clients



    #对选中的客户端模型进行训练，并将其更新为全局模型的最新参数
    def client_update(self,chosen_clients):
        self.losses=[]# update nn
        for j in chosen_clients:
            self.nns[j]= train(self.args,self.nns[j], self.nn, self.mu)
            
            
    def global_test(self, file_name):
        model = self.nn
        model.eval()
        test_loss, recall_test, percision_test, F1_test=test(self.args, model,file_name)
        return test_loss, recall_test, percision_test, F1_test

    