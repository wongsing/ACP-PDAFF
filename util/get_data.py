# -*- coding: utf-8 -*-
# @Time    : 2023/6/28 11:05
# @Author  : WANG Xinyi
# @Email   : wangxinyi_tazr@stu.ynu.edu.cn
# @IDE     : PyCharm
# @FileName: get_data.py

import numpy as np
import pandas as pd
import os
import json
import re
import math
import platform
from itertools import product
import torch
import torch.nn.utils.rnn as rnn_utils



def parse_stream(f, comment=b'#'):
    name = None
    sequence = []
    for line in f:
        if line.startswith(comment):
            continue
        line = line.strip()
        if line.startswith(b'>'):
            if name is not None:
                yield name, b''.join(sequence)
            name = line[1:]
            sequence = []
        else:
            sequence.append(line.upper())
    if name is not None:
        yield name, b''.join(sequence)


def fasta2csv(inFasta):
    FastaRead = pd.read_csv(inFasta, header=None)

    seqNum = int(FastaRead.shape[0] / 2)
    csvFile = open(os.path.join("../data/dataset", "testFasta.csv"), "w")
    csvFile.write("PID,Seq\n")

    for i in range(seqNum):
        csvFile.write(str(FastaRead.iloc[2 * i, 0]) + "," + str(FastaRead.iloc[2 * i + 1, 0]) + "\n")

    csvFile.close()
    TrainSeqLabel = pd.read_csv(os.path.join( "../data/dataset", "testFasta.csv"), header=0)
    path = os.path.join("../data/dataset", "testFasta.csv")
    if os.path.exists(path):
        os.remove(path)
    return TrainSeqLabel

def get_sequence(filename):
    inFastaTest = os.path.join("../data/dataset", filename)
    # print('infastatest:',inFastaTest)
    mainTest = fasta2csv(inFastaTest)
    # print('mainTest:',mainTest)
    i = 0
    mainTest["Tags"] = mainTest["Seq"]
    for pid in mainTest["PID"]:
        mainTest["Tags"][i] = pid[len(pid) - 1]
        if mainTest["Tags"][i] == "1":
            mainTest["Tags"][i] = 1
        else:
            mainTest["Tags"][i] = 0
        i = i + 1
    ACP_y_test = mainTest["Tags"].values
    ACP_y_test_ = np.array([np.array(i) for i in ACP_y_test])
    ACP_y_test_ = torch.tensor(ACP_y_test_)
    # label = torch.tensor(pro['label'].values)
    x_test = {}
    protein_index = 1
    for line in mainTest["Seq"]:
        x_test[protein_index] = line
        protein_index = protein_index + 1
    maxlen_test = max(len(x) for x in x_test.values())

    maxlen = maxlen_test

    ACP_x_test = []
    for seq in x_test.values():
        ACP_x_test.append(seq)

    return ACP_x_test,ACP_y_test_

def get_vocab(base,n):
    result = []
    for i in product(base,repeat=n):
        result.append(''.join(i))
    return result

def get_onehot(sequence,vocab):

    char_to_int = dict((c, i) for i, c in enumerate(vocab))
    int_to_char = dict((i, c) for i, c in enumerate(vocab))

    integer_encoded = [char_to_int[char] for char in sequence]

    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(vocab))]
        letter[value] = 1
        onehot_encoded.append(letter)

    return onehot_encoded

base = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def genData(lines):
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16,'T': 17,'W': 18, 'Y': 19, 'V': 20}

    long_pep_counter = 0
    pep_codes = []
    pep_seq = []
    max_seq_len = 100

    for pep in lines:
        input_seq = ' '.join(pep)
        input_seq = re.sub(r"[UZOB]", "X", input_seq)
        pep_seq.append(input_seq)
        if not len(pep) > max_seq_len:
            current_pep = []
            for aa in pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(torch.tensor(current_pep))
        else:
            long_pep_counter += 1
    print("length > 50 :", long_pep_counter)
    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)
    return data, pep_seq

def load_tsv_format_data(file, skip_head=True):
    sequences = []
    labels = []
    filename = '../data/dataset/'+file

    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            sequences.append(list[2])
            labels.append(int(list[1]))

    return sequences, labels

