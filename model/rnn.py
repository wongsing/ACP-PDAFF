# -*- coding: utf-8 -*-
# @Time    : 2023/6/28 11:05
# @Author  : WANG Xinyi
# @Email   : wangxinyi_tazr@stu.ynu.edu.cn
# @IDE     : PyCharm
# @FileName: rnn.py

import torch
from torch import nn
import numpy as np
import pandas as pd
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(1024, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 1024)
        self.mse = nn.MSELoss()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        # print('z:',z)
        return self.decode(z), mu, logvar

class DAFF(nn.Module):
    '''
    AFF,Attentional Feature Fusion
    '''

    def __init__(self, channels=1024, r=4):
        super(DAFF, self).__init__()
        inter_channels = int(channels // r)

        # loacl attention
        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        # global attention
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        """the summation of dual-channel fearures """
        xa = x + residual

        xl = self.local_att(x)
        xg = self.local_att(residual)
        xa = self.local_att(xa)
        xlg = xl + xg+xg

        """original local attentional features fusion(OLAFF)"""
        wei_olaff = self.sigmoid(xa)

        """PDAFF we proposed"""
        wei = self.sigmoid(xlg)

        xo = x * wei + residual * (1 - wei)
        return xo


class newModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vae = VAE()

        self.aff = DAFF()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=25, nhead=5,dim_feedforward=64,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2,)
        self.block1 = nn.Sequential(
                                    nn.Linear(5175,1024),
                                    # nn.BatchNorm1d(2048),
                                    # nn.Linear(2048, 1024),
                                    )

        self.block2 = nn.Sequential(
            nn.Linear(1024, 640),
            nn.BatchNorm1d(640),
            nn.ReLU6(),
            nn.Linear(640,2)
        )

    def forward(self, x,pep):
        x = x.type(torch.float32)
        output = self.transformer_encoder(x)
        output = output.reshape(output.shape[0], -1)
        return self.block1(output)



    def get_logits(self, x,pep):
        with torch.no_grad():
            output = self.forward(x,pep)

        """VAE"""
        recon_x ,p_x, q_z = self.vae(pep)

        output1 = output.view(output.shape[0],output.shape[1],1)
        pep_x1 = pep.view(pep.shape[0], pep.shape[1],1)

        """concatenation"""
        # output2 = torch.cat([output1,pep_x1],dim=1)
        """summation"""
        # output3 = output1+pep_x1
        """dual-channel attentional feature fusion"""
        output4 = self.aff(output1,pep_x1)

        output4 = output4.reshape(output4.shape[0], -1)
        logits = self.block2(output4)
        # return logits,recon_x ,p_x, q_z
        return logits