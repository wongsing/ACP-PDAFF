# -*- coding: utf-8 -*-
# @Time    : 2023/6/28 11:05
# @Author  : WANG Xinyi
# @Email   : wangxinyi_tazr@stu.ynu.edu.cn
# @IDE     : PyCharm
# @FileName: get_feature.py

import os
import platform
import re
import math
import pandas as pd
import numpy as np
from util import get_data
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from numpy import array,argmax,linalg as la

#论文用的keras，这里得转成tensor
def get_pad_oe(x_train_oe):
    x_train_ = pad_sequence(x_train_oe, batch_first=True,padding_value=0)
    # x_test_ = np.array(pad_sequence(x_test_oe, batch_first=True))
    return x_train_

inFastaTrain = "../dataset/ACP20mainTrain.fasta"
inFastaTest = "../dataset/ACP20mainTest.fasta"

import re
import sys
def readFasta(file):
    if os.path.exists(file) == False:
        print('Error: "' + file + '" dOEs not exist.')
        sys.exit(1)

    with open(file) as f:
        records = f.read()

    if re.search('>', records) == None:
        print('The input file seems not in fasta format.')
        sys.exit(1)

    records = records.split('>')[1:]
    myFasta = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
        myFasta.append([name, sequence])

    return myFasta





def BINARY(sequence):
    encodings = []
    # print(sequence)
    for seq in sequence:
        code = []
        # print(seq)
        for aa in seq:
            if aa == '-':
                code.append([0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0])
            if aa == 'A':
                code.append([1, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0])
            if aa == 'C':
                code.append([0, 1, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0])
            if aa == 'D':
                code.append([0, 0, 1, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0])
            if aa == 'E':
                code.append([0, 0, 0, 1,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0])
            if aa == 'F':
                code.append([0, 0, 0, 0,1, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0])
            if aa == 'G':
                code.append([0, 0, 0, 0,0, 1, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0])
            if aa == 'H':
                code.append([0, 0, 0, 0,0, 0, 1, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0])
            if aa == 'I':
                code.append([0, 0, 0, 0,0, 0, 0, 1,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0])
            if aa == 'K':
                code.append([0, 0, 0, 0,0, 0, 0, 0,1, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0])
            if aa == 'L':
                code.append([0, 0, 0, 0,0, 0, 0, 0,0, 1, 0, 0,0, 0, 0, 0,0, 0, 0, 0])
            if aa == 'M':
                code.append([0, 0, 0, 0,0, 0, 0, 0,0, 0, 1, 0,0, 0, 0, 0,0, 0, 0, 0])
            if aa == 'N':
                code.append([0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 1,0, 0, 0, 0,0, 0, 0, 0])
            if aa == 'P':
                code.append([0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,1, 0, 0, 0,0, 0, 0, 0])
            if aa == 'Q':
                code.append([0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 1, 0, 0,0, 0, 0, 0])
            if aa == 'R':
                code.append([0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 1, 0,0, 0, 0, 0])
            if aa == 'S':
                code.append([0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 1,0, 0, 0, 0])
            if aa == 'T':
                code.append([0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,1, 0, 0, 0])
            if aa == 'V':
                code.append([0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 1, 0, 0])
            if aa == 'W':
                code.append([0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 1, 0])
            if aa == 'Y':
                code.append([0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 1])
        # print(code)
        encodings.append(code)
    return encodings

def get_pp(sequence):
    group = {
        'uncharger': 'STCPNQ',
        'negativecharger': 'DE',
        'postivecharger': 'KRH',
        'aromatic': 'FYW',
        'alphaticr': 'GAVLMI'
    }
    AA = 'ARNDCQEGHILKMFPSTWYV'
    groupKey = group.keys()
    code = []

    for p1 in range(len(sequence)):
        val = [0] * 5
        i = 0
        for key in groupKey:
            for aa in group[key]:
                if sequence[p1] == aa :
                    val[i] = 1
                    break
            i=i+1
        code.append(val)

    return code

def get_one(x_train,is_one):
    x_train_one = []
    for x in x_train:
        if is_one:
            one_hot = BINARY(x)
        else:
            one_hot = get_pp(x)
        one_hot = torch.tensor(one_hot)
        # print(one_hot.shape)
        one_hot = torch.squeeze(one_hot,dim=1)
        # print(one_hot.shape)
        x_train_one.append(one_hot)
        # print(np.array(x_train_one).shape)
    x_train_one = get_pad_oe(x_train_one)
    # print(x_train_one.shape)
    return x_train_one

"""NT-padding/CT-padding"""
def get_one_new(x_train,is_one,max_len,padding):
    x_test_new = []
    for i in x_train:
        if len(i) > max_len:
            # print(len(i))
            x_test_new.append(i[:max_len])
        else:
            x_test_new.append(i)

    x_train_one = []
    for x in x_test_new:
        if is_one:
            one_hot = BINARY(x)
        else:
            one_hot = get_pp(x)
        one_hot = torch.tensor(one_hot)
        # print(one_hot.shape)
        one_hot = torch.squeeze(one_hot,dim=1)
        if padding == 'NT':
        # print('before:',one_hot,one_hot.shape)
            pad_len = max_len - one_hot.shape[0]
            padding_vector = torch.zeros((pad_len, one_hot.shape[1]))
            one_hot = torch.cat([padding_vector,one_hot],dim=0)
            # one_hot = torch.tensor(one_hot)
            # print('after:',one_hot,one_hot.shape)
        x_train_one.append(one_hot)
        # print(np.array(x_train_one).shape)
    x_train_one = torch.stack(x_train_one)
    if padding == 'CT':
        x_train_one = pad_sequence(x_train_one, batch_first=True, padding_value=0)
    return x_train_one

def readFasta(file):
    with open(file) as f:
         records=f.read()
    if re.search('>',records)==None:
       print('error in fasta format')
       sys.exit(1)
    records=records.split('>')[1:]
    myFasta=[]
    for fasta in records:
        array=fasta.split('\n')
        name, sequence=array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
        myFasta.append(sequence)
    return myFasta



"""test"""
# x_train = readFasta(inFastaTrain)
# x_test = readFasta(inFastaTest)
# # print('zaiheli:',x_train)
# x_train_one = get_one(x_train)
# x_test_one = get_one(x_test)
# # print(np.array(x_test_one).shape,x_test_one)
#
# one_data = pd.DataFrame(data=x_test_one)
# one_data.to_csv('test_one_data.csv')
# x_train_pp = get_pp(x_train)
# x_test_pp = get_pp(x_test)
# print(np.array(x_train_pp).shape,np.array(x_test_pp).shape)
# print(x_test_pp)




# a = [0] * 22
# print(a)