# -*- coding: utf-8 -*-
# @Time    : 2023/6/28 11:05
# @Author  : WANG Xinyi
# @Email   : wangxinyi_tazr@stu.ynu.edu.cn
# @IDE     : PyCharm
# @FileName: pretrain.py
import re
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import json
import pandas as pd

from transformers import T5Tokenizer, T5EncoderModel, BertTokenizer, BertModel

from util import get_data

class MyDataSet(Data.Dataset):
    def __init__(self, data, label, seq):
        self.data = data
        self.label = label
        self.seq = seq

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.seq[idx]

def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []

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

if __name__ == '__main__':

    device = 'cpu'

    """dataset is in fasta format"""
    # x_test,y_test = get_data.get_sequence('ACP20alternateTest.fasta') #342
    # x_train,y_train = get_data.get_sequence('ACP20alternateTrain.fasta') #1376
    """dataset is in csv format"""
    x_train,y_train  = load_tsv_format_data('data/dataset/ACP_FL_train_500.tsv')
    x_test,y_test = load_tsv_format_data('data/dataset/ACP_FL_test_164.tsv')

    print(np.array(x_train).shape,np.array(y_train).shape)
    print(np.array(x_test).shape,np.array(y_test).shape)
    seq = x_train + x_test
    print(np.array(seq).shape)

    # sequence_examples = ["PRTEINO", "SEQWENCE"]
    # # this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq]
    print(sequence_examples)

#Unsupervised learning
    #prot-bert-bfd
    tokenizer = BertTokenizer.from_pretrained("./prot_bert_bfd", do_lower_case=False)
    bert = BertModel.from_pretrained("./prot_bert_bfd")
    seq2vec = dict()
    for pep in sequence_examples:
        pep_str = "".join(pep)
        pep_text = tokenizer.tokenize(pep_str)
        pep_tokens = tokenizer.convert_tokens_to_ids(pep_text)
        tokens_tensor = torch.tensor([pep_tokens])
        with torch.no_grad():
            encoder_layers = bert(tokens_tensor)
            out_ten = torch.mean(encoder_layers.last_hidden_state, dim=1)
            out_ten = out_ten.numpy().tolist()
            seq2vec[pep] = out_ten

    with open('data/pretrain/seq2vec_ACP_FL1.emb', 'w') as g:
        g.write(json.dumps(seq2vec))


