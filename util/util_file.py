# -*- coding: utf-8 -*-
# @Time    : 2023/6/28 11:05
# @Author  : WANG Xinyi
# @Email   : wangxinyi_tazr@stu.ynu.edu.cn
# @IDE     : PyCharm
# @FileName: util_file.py
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn
import torch.utils.data as Data
import numpy as np
import os
import itertools
from collections import Counter
import re,math,platform

def toNum(l):
    l = [float(i) for i in l]
    return l

def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []
    # CLS_labels = []
    # pssm = []
    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            l = line.split('\t')
            # if len(l[2]) <= 1000:
            sequences.append(l[0])
            # print(l[0])
            label = list(l[1])
            label = [int(i) for i in label]
            labels.append(label)
            # CLS_labels.append(int(l[2]))
            # l3 = l[3].split(',')
            # l = [toNum(i.split()) for i in l3][:-1]  # seq_len * 20
            # pssm.append(l)                           # seq_num * seq_len * 20

    return sequences, labels

#获取交叉验证数据集
def get_kfold_data(rep_dir,pro_label_dir):
    rep_all_pd = pd.read_csv(rep_dir)
    pro = pd.read_csv(pro_label_dir)
    label = torch.tensor(pro['label'].values)
    head, tail = os.path.split(pro_label_dir)
    # print('tail:',tail)
    trP = tail.split('trP')[1].split('_')[0]
    trN = tail.split('trN')[1].split('_')[0]
    vaP = tail.split('VaP')[1].split('_')[0]
    vaN = tail.split('VaN')[1].split('_')[0]
    teP = tail.split('TeP')[1].split('_')[0]
    teN = tail.split('TeN')[1].split('_')[0]
    data = torch.tensor(rep_all_pd.values)
    # print('data:',data)
    print('where are u?', trP, trN, vaP, vaN, teP, teN)
    print(data.shape, label.shape)
    print(label.shape, data.shape)
    train_data, train_label = data[:int(trP) + int(trN)+int(vaP) + int(vaN)].double(), label[:int(trP) + int(trN)+int(vaP) + int(vaN)]
    # valid_data = data[int(trP) + int(trN):int(trP) + int(trN)+int(vaP) + int(vaN)].double()
    # valid_label = label[int(trP) + int(trN):int(trP) + int(trN) + int(vaP) + int(vaN)]
    test_data, test_label = data[int(trP) + int(trN)+int(vaP) + int(vaN):].double(), label[int(trP) + int(trN)+int(vaP) + int(vaN):]
    print('train_data:',len(train_data),len(train_label))
    # print('validation_data:', len(valid_data),len(valid_label))
    print('test_data:', len(test_data),len(test_label))
    # LOSS_WEIGHT_POSITIVE = math.sqrt((int(trP)+int(trN)) / (2.0 * int(trP)) )
    # LOSS_WEIGHT_NEGATIVE = math.sqrt((int(trP)+int(trN)) / (2.0 * int(trN)) )
    LOSS_WEIGHT_POSITIVE = (int(trP) + int(trN)) / (2.0 * int(trP))
    LOSS_WEIGHT_NEGATIVE = (int(trP) + int(trN)) / (2.0 * int(trN))
    # https://towardsdatascience.com/deep-learning-with-weighted-cross-entropy-loss-on-imbalanced-tabular-data-using-fastai-fe1c009e184c
    soft_max = nn.Softmax(dim=1)
    # class_weights=torch.FloatTensor([w_0, w_1]).cuda()
    # weig=torch.FloatTensor([LOSS_WEIGHT_NEGATIVE,LOSS_WEIGHT_POSITIVE]).double().cuda()
    weig = torch.FloatTensor([LOSS_WEIGHT_NEGATIVE, LOSS_WEIGHT_POSITIVE])
    # train_data,train_label=genData("./train_peptide.csv",260)
    # test_data,test_label=genData("./test_peptide.csv",260)

    train_dataset = Data.TensorDataset(train_data, train_label)
    test_dataset = Data.TensorDataset(test_data, test_label)
    # valid_dataset = Data.TensorDataset(valid_data,valid_label)
    batch_size = 256
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_iter,test_iter,weig

# 数据集1为训练集，数据集2为测试集
def get_cross_data(rep_dir1,pro_label_dir1,rep_dir2,pro_label_dir2):
    rep_all_pd1 = pd.read_csv(rep_dir1)
    pro1 = pd.read_csv(pro_label_dir1)
    label1 = torch.tensor(pro1['label'].values)
    head1, tail1 = os.path.split(pro_label_dir1)
    # print('tail:',tail)
    trP1 = tail1.split('trP')[1].split('_')[0]
    trN1 = tail1.split('trN')[1].split('_')[0]
    vaP1 = tail1.split('VaP')[1].split('_')[0]
    vaN1 = tail1.split('VaN')[1].split('_')[0]
    teP1 = tail1.split('TeP')[1].split('_')[0]
    teN1 = tail1.split('TeN')[1].split('_')[0]
    data1 = torch.tensor(rep_all_pd1.values)
    # print('data:',data)

    rep_all_pd2 = pd.read_csv(rep_dir2)
    pro2 = pd.read_csv(pro_label_dir2)
    label2 = torch.tensor(pro2['label'].values)
    head2, tail2 = os.path.split(pro_label_dir2)

    trP2 = tail2.split('trP')[1].split('_')[0]
    trN2 = tail2.split('trN')[1].split('_')[0]
    vaP2 = tail2.split('VaP')[1].split('_')[0]
    vaN2 = tail2.split('VaN')[1].split('_')[0]
    teP2 = tail2.split('TeP')[1].split('_')[0]
    teN2 = tail2.split('TeN')[1].split('_')[0]
    data2 = torch.tensor(rep_all_pd2.values)

    train_data1, train_label1 = data1[:int(trP1) + int(trN1)+int(vaP1) + int(vaN1)].double(), label1[:int(trP1) + int(trN1)+int(vaP1) + int(vaN1)]
    valid_data1 = data1[int(trP1) + int(trN1):int(trP1) + int(trN1)+int(vaP1) + int(vaN1)].double()
    valid_label1 = label1[int(trP1) + int(trN1):int(trP1) + int(trN1) + int(vaP1) + int(vaN1)]
    test_data2, test_label2 = data2[int(trP2) + int(trN2)+int(vaP2) + int(vaN2):].double(), label2[int(trP2) + int(trN2)+int(vaP2) + int(vaN2):]
    print('train_data:',len(train_data1),len(train_label1))
    print('validation_data:', len(valid_data1),len(valid_label1))

    print('test_data:', len(test_data2),len(test_label2))
    # LOSS_WEIGHT_POSITIVE = math.sqrt((int(trP)+int(trN)) / (2.0 * int(trP)) )
    # LOSS_WEIGHT_NEGATIVE = math.sqrt((int(trP)+int(trN)) / (2.0 * int(trN)) )
    LOSS_WEIGHT_POSITIVE = (int(trP1) + int(trN1)) / (2.0 * int(trP1))
    LOSS_WEIGHT_NEGATIVE = (int(trP1) + int(trN1)) / (2.0 * int(trN1))
    # https://towardsdatascience.com/deep-learning-with-weighted-cross-entropy-loss-on-imbalanced-tabular-data-using-fastai-fe1c009e184c
    soft_max = nn.Softmax(dim=1)
    # class_weights=torch.FloatTensor([w_0, w_1]).cuda()
    # weig=torch.FloatTensor([LOSS_WEIGHT_NEGATIVE,LOSS_WEIGHT_POSITIVE]).double().cuda()
    weig = torch.FloatTensor([LOSS_WEIGHT_NEGATIVE, LOSS_WEIGHT_POSITIVE])
    # train_data,train_label=genData("./train_peptide.csv",260)
    # test_data,test_label=genData("./test_peptide.csv",260)

    train_dataset = Data.TensorDataset(train_data1, train_label1)
    test_dataset = Data.TensorDataset(test_data2, test_label2)
    valid_dataset = Data.TensorDataset(valid_data1,valid_label1)
    batch_size = 256
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_iter,test_iter,valid_iter,weig

def read_fasta(rep_dir,pro_label_dir):
    rep_all_pd=pd.read_csv(rep_dir)
    pro=pd.read_csv(pro_label_dir)
    label=torch.tensor(pro['label'].values)
    head,tail=os.path.split(pro_label_dir)
    trP=tail.split('trP')[1].split('_')[0]
    trN=tail.split('trN')[1].split('_')[0]
    vaP=tail.split('VaP')[1].split('_')[0]
    vaN=tail.split('VaN')[1].split('_')[0]
    teP=tail.split('TeP')[1].split('_')[0]
    teN=tail.split('TeN')[1].split('_')[0]
    data=torch.tensor(rep_all_pd.values)
    # print('where are u?',trP,trN,vaP,vaN,teP,teN)
    # print(data.shape,label.shape)
    # print(label.shape,data.shape)
    train_data,train_label=data[:int(trP)+int(trN)].double(),label[:int(trP)+int(trN)]
    val_data, val_label = data[int(trP) + int(trN):int(trP) + int(trN)+int(vaP) + int(vaN)].double(), label[:int(trP) + int(trN)+int(vaP) + int(vaN)]
    test_data,test_label=data[int(trP)+int(trN):-int(teP)-int(teN)].double(),label[int(trP)+int(trN):-int(teP)-int(teN)]
    # LOSS_WEIGHT_POSITIVE = math.sqrt((int(trP)+int(trN)) / (2.0 * int(trP)) )
    # LOSS_WEIGHT_NEGATIVE = math.sqrt((int(trP)+int(trN)) / (2.0 * int(trN)) )
    LOSS_WEIGHT_POSITIVE = (int(trP)+int(trN)) / (2.0 * int(trP))
    LOSS_WEIGHT_NEGATIVE = (int(trP)+int(trN)) / (2.0 * int(trN))
    # https://towardsdatascience.com/deep-learning-with-weighted-cross-entropy-loss-on-imbalanced-tabular-data-using-fastai-fe1c009e184c
    soft_max=nn.Softmax(dim=1)
    # class_weights=torch.FloatTensor([w_0, w_1]).cuda()
    # weig=torch.FloatTensor([LOSS_WEIGHT_NEGATIVE,LOSS_WEIGHT_POSITIVE]).double().cuda()
    weig=torch.FloatTensor([LOSS_WEIGHT_NEGATIVE,LOSS_WEIGHT_POSITIVE]).double()
    # train_data,train_label=genData("./train_peptide.csv",260)
    # test_data,test_label=genData("./test_peptide.csv",260)


def get_seqence(file):
    f = open(file)
    documents = f.readlines()
    string = ""
    flag = 0
    fea=[]
    for document in documents:
        if document.startswith(">") and flag == 0:
            flag = 1
            continue
        elif document.startswith(">") and flag == 1:
            string=string.upper()
            fea.append(string)
            string = ""
        else:
            string += document
            string = string.strip() #剥离空白字符
            string=string.replace(" ", "")
    fea.append(string)
    f.close()
    return fea
