# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 23:41:17 2021

@author: qingyi
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASU_NN(torch.nn.Module):
    
    def __init__(self,att_index,n_hidden_1 = 64, n_hidden_2 = 128,
                 n_layer_1 = 3, n_layer_2 = 3, n_output = 6, 
                 device = torch.device('cpu')):
        
        super(ASU_NN, self).__init__()
        self.att_index = att_index
        self.n_layer_1 = n_layer_1
        self.n_layer_2 = n_layer_2
        self.n_output = n_output
        self.device = device
        self.fc_input = nn.ModuleDict({})
        self.fc_middle = nn.ModuleDict({})
        self.fc_merge = nn.ModuleDict({})
        self.fc_middle2 = nn.ModuleDict({})
        self.out = nn.ModuleDict({})

        for key in att_index:
            n_alt_feature = len(att_index[key])
            self.fc_input.update({key:torch.nn.Linear(n_alt_feature, n_hidden_1)})
            
            
            for j in range(n_layer_1):
                self.fc_middle.update({key+'_'+str(j+1):torch.nn.Linear(n_hidden_1, n_hidden_1)})
            
            
            if 'x_' in key:
                '''
                if 'z' in att_index:
                    self.fc_merge.update({key:torch.nn.Linear(n_hidden_1*2, n_hidden_2)})
                else:
                    self.fc_merge.update({key:torch.nn.Linear(n_hidden_1, n_hidden_2)})
                
                for j in range(n_layer_1):
                    self.fc_middle2.update({key+'_'+str(j+1):torch.nn.Linear(n_hidden_2, n_hidden_2)})
                '''
                self.out.update({key: torch.nn.Linear(n_hidden_1, 1)})
        
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        #
        #num_samples = list(x.size())[0]
        x_temp = {}
        for key in self.att_index:
            x_temp[key] = F.relu(self.fc_input[key](x[:,self.att_index[key]]))     
            
            for j in range(self.n_layer_1):
                x_temp[key] = F.relu(self.fc_middle[key+"_"+str(j+1)](x_temp[key]))
            
        '''
        if 'z' in x_temp.keys():                
            x_temp2 = {}
            for key in self.att_index:
                if 'x_' in key:
                    x_temp2[key] = torch.cat((x_temp[key], x_temp['z']),dim=1)
        else:
            x_temp2 = x_temp
            
        for key in self.att_index:
            if 'x_' in key:
                x_temp2[key] = F.relu(self.fc_merge[key](x_temp2[key]))
            # if key == 'x_1':
            #     print(key, x_temp2[key][:2,:5].detach().numpy())
        
        for key in self.att_index:
            if 'x_' in key:
                for j in  range(self.n_layer_2):
                    x_temp2[key] = self.fc_middle2[key+"_"+str(j+1)](x_temp2[key])
            #if key == 'x_1':
            #    print(key, x_temp2[key][:2,:5].detach().numpy())
        '''
        # out
        for key in self.att_index:
            if 'x_' in key:
                x_temp[key] = self.out[key](x_temp[key])
                #print(key, x_temp2[key][:2,:5].detach().numpy())
                
        #[torch.zeros(num_samples,1).to(self.device)] + 
        x_out = torch.cat([x_temp[key] for key in x_temp],dim = 1) 
        #print(x_out.shape)
        # print(x_out[:5,:])
        #mask = x[:,np.arange(0,30,6)]>0
        x_out = F.softmax(x_out, dim=1)
        return x_out


class ASU_NN_RL(torch.nn.Module):

    def __init__(self, att_index, n_hidden_1=64, n_hidden_2=128,
                 n_layer_1=3, n_layer_2=3, n_output=6,
                 device=torch.device('cpu')):

        super(ASU_NN_RL, self).__init__()
        self.att_index = att_index
        self.n_layer_1 = n_layer_1
        self.n_layer_2 = n_layer_2
        self.n_output = n_output
        self.device = device
        self.fc_input = nn.ModuleDict({})
        self.fc_middle = nn.ModuleDict({})
        self.fc_merge = nn.ModuleDict({})
        self.fc_middle2 = nn.ModuleDict({})
        self.out = nn.ModuleDict({})

        for key in att_index:
            n_alt_feature = len(att_index[key])
            self.fc_input.update({key: torch.nn.Linear(n_alt_feature, n_hidden_1)})

            for j in range(n_layer_1):
                self.fc_middle.update({key + '_' + str(j + 1): torch.nn.Linear(n_hidden_1, n_hidden_1)})

            if 'x_' in key:
                '''
                if 'z' in att_index:
                    self.fc_merge.update({key:torch.nn.Linear(n_hidden_1*2, n_hidden_2)})
                else:
                    self.fc_merge.update({key:torch.nn.Linear(n_hidden_1, n_hidden_2)})

                for j in range(n_layer_1):
                    self.fc_middle2.update({key+'_'+str(j+1):torch.nn.Linear(n_hidden_2, n_hidden_2)})
                '''
                self.out.update({key: torch.nn.Linear(n_hidden_1, 1)})

        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        #
        # num_samples = list(x.size())[0]
        x_temp = {}
        for key in self.att_index:
            x_temp[key] = F.relu(self.fc_input[key](x[:, self.att_index[key]]))

            for j in range(self.n_layer_1):
                x_temp[key] = F.relu(self.fc_middle[key + "_" + str(j + 1)](x_temp[key]))

        '''
        if 'z' in x_temp.keys():                
            x_temp2 = {}
            for key in self.att_index:
                if 'x_' in key:
                    x_temp2[key] = torch.cat((x_temp[key], x_temp['z']),dim=1)
        else:
            x_temp2 = x_temp

        for key in self.att_index:
            if 'x_' in key:
                x_temp2[key] = F.relu(self.fc_merge[key](x_temp2[key]))
            # if key == 'x_1':
            #     print(key, x_temp2[key][:2,:5].detach().numpy())

        for key in self.att_index:
            if 'x_' in key:
                for j in  range(self.n_layer_2):
                    x_temp2[key] = self.fc_middle2[key+"_"+str(j+1)](x_temp2[key])
            #if key == 'x_1':
            #    print(key, x_temp2[key][:2,:5].detach().numpy())
        '''
        # out
        for key in self.att_index:
            if 'x_' in key:
                x_temp[key] = self.out[key](x_temp[key])
                # print(key, x_temp2[key][:2,:5].detach().numpy())

        # [torch.zeros(num_samples,1).to(self.device)] +
        Q_value = torch.cat([x_temp[key] for key in x_temp], dim=1)
        # print(x_out.shape)
        # print(x_out[:5,:])
        # mask = x[:,np.arange(0,30,6)]>0
        #x_out = F.softmax(x_out, dim=1)

        #Q_value

        return Q_value


class ASU_NN_same_para(torch.nn.Module):

    def __init__(self, att_index, n_hidden_1=64, n_hidden_2=128,
                 n_layer_1=3, n_layer_2=3, n_output=6,
                 device=torch.device('cpu')):

        super(ASU_NN_same_para, self).__init__()
        self.att_index = att_index
        self.n_layer_1 = n_layer_1
        self.n_layer_2 = n_layer_2
        self.n_output = n_output
        self.device = device
        self.fc_input = nn.ModuleDict({})
        self.fc_middle = nn.ModuleDict({})
        self.fc_merge = nn.ModuleDict({})
        self.fc_middle2 = nn.ModuleDict({})
        self.out = nn.ModuleDict({})

        key = 'only_one'

        n_alt_feature = len(att_index['x_1'])
        self.fc_input.update({key: torch.nn.Linear(n_alt_feature, n_hidden_1)})

        for j in range(n_layer_1):
            self.fc_middle.update({key + '_' + str(j + 1): torch.nn.Linear(n_hidden_1, n_hidden_1)})


        #self.out.update({key: torch.nn.Linear(n_hidden_1, 1)})

        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        #
        # num_samples = list(x.size())[0]
        x_temp = {}
        key_net = 'only_one'
        for key in self.att_index:
            x_temp[key] = F.relu(self.fc_input[key_net](x[:, self.att_index[key]]))
            for j in range(self.n_layer_1):
                x_temp[key] = F.relu(self.fc_middle[key_net + "_" + str(j + 1)](x_temp[key]))


        x_out = torch.cat([x_temp[key] for key in x_temp], dim=1)
        x_out = F.softmax(x_out, dim=1)
        return x_out
