# -*- coding: utf-8 -*-
# @Time    : 2023/6/28 11:05
# @Author  : WANG Xinyi
# @Email   : wangxinyi_tazr@stu.ynu.edu.cn
# @IDE     : PyCharm
# @FileName: ACP_PDAFF_main.py
import json
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from configuration import config as cf
from util import util_metric, get_feature
from train.visualization import dimension_reduction, penultimate_feature_visulization
from model import rnn
from util import get_data
from sklearn.metrics import (auc)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import pickle
import seaborn as sns
import random
import pandas as pd
import torch.utils.data as Data
from scipy import interp
from train.model_operation import save_model

def draw_figure_CV(config, fig_name):
    sns.set(style="darkgrid")
    plt.figure(22, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i, e in enumerate(train_acc_record):
        train_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(train_loss_record):
        # train_loss_record[i] = e.cpu().detach()
        # print(e)
        train_loss_record[i] = e.cpu().detach()

    for i, e in enumerate(valid_acc_record):
        valid_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(valid_loss_record):
        # valid_loss_record[i] = e.cpu().detach()
        valid_loss_record[i] = e

    plt.subplot(2, 2, 1)
    plt.title("Train Acc Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_log_interval, train_acc_record)
    plt.subplot(2, 2, 2)
    plt.title("Train Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_log_interval, train_loss_record)
    plt.subplot(2, 2, 3)
    plt.title("Validation Acc Curve", fontsize=23)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_valid_interval, valid_acc_record)
    plt.subplot(2, 2, 4)
    plt.title("Validation Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_valid_interval, valid_loss_record)

    plt.savefig(config.result_folder + '/' + fig_name + '.png')
    plt.show()

def draw_figure_train_test(config, fig_name):
    # sns.set(style="darkgrid")darkgrid
    plt.figure(22, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i, e in enumerate(train_acc_record):
        train_acc_record[i] = e.cpu().detach()
        # print(train_acc_record[i])
    for i, e in enumerate(train_loss_record):
        train_loss_record[i] = e.cpu().detach()
        # train_loss_record[i] = e
        # print(train_loss_record[i])
    for i, e in enumerate(test_acc_record):
        test_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(test_loss_record):
        # test_loss_record[i] = e.cpu().detach()
        test_loss_record[i] = e

    plt.subplot(2, 2, 1)
    plt.title("Train Acc Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_log_interval, train_acc_record)
    # plt.plot(step_test_interval, test_acc_record,color='c')
    plt.subplot(2, 2, 2)
    plt.title("Train Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_log_interval, train_loss_record)
    plt.subplot(2, 2, 3)
    plt.title("Test Acc Curve", fontsize=23)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_test_interval, test_acc_record)
    plt.subplot(2, 2, 4)
    plt.title("Test Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_test_interval, test_loss_record)

    plt.savefig(config.result_folder + '/' + fig_name + '.png')
    plt.show()

def get_loss(logits, label, criterion):
    loss = criterion(logits, label)
    loss = loss.float()
    loss = (loss - config.b).abs() + config.b
    return loss

"""VAE loss"""
def loss_function_original( recon_x, x, mu, logvar,criterion):
    BCE = criterion(recon_x, x)
    # print('BCE:',BCE)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print("KLD:",KLD)
    return BCE + KLD

def periodic_test(test_iter, model, criterion, config, sum_epoch):
    print('#' * 60 + 'Periodic Test' + '#' * 60)
    test_metric, test_loss, test_repres_list, test_label_list, \
    test_roc_data, test_prc_data = model_eval(test_iter, model, criterion, config)

    print('test current performance')
    print('[ACC,\t\tPrecision,\t\tSensitivity,\tSpecificity,\t\tF1,\t\tAUC,\t\tMCC,\t\tAP,\t\tTP,\t\tFP,\t\t\tTN,\t\t\tFN]')
    plmt = test_metric.numpy()
    print('%.5g\t\t' % plmt[0], '%.5g\t\t' % plmt[1], '%.5g\t\t' % plmt[2], '%.5g\t\t' % plmt[3], '%.5g\t' % plmt[4],'%.5g\t\t' % plmt[5],
          '%.5g\t\t' % plmt[6], '%.5g\t\t' % plmt[7], '  %.5g\t\t' % plmt[8], '  %.5g\t\t' % plmt[9], ' %.5g\t\t' % plmt[10],' %.5g\t\t' % plmt[11])
    print('#' * 60 + 'Over' + '#' * 60)

    step_test_interval.append(sum_epoch)
    test_acc_record.append(test_metric[0])
    test_loss_record.append(test_loss)

    return test_metric, test_loss, test_repres_list, test_label_list


def periodic_valid(valid_iter, model, criterion, config, sum_epoch):
    print('#' * 60 + 'Periodic Validation' + '#' * 60)

    valid_metric, valid_loss, valid_repres_list, valid_label_list, \
    valid_roc_data, valid_prc_data = model_eval(valid_iter, model, criterion, config)

    print('validation current performance')
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC,\tAP,\tTP,\tFP,\tTN,\tFN]')
    print(valid_metric.numpy())
    print('#' * 60 + 'Over' + '#' * 60)

    step_valid_interval.append(sum_epoch)
    valid_acc_record.append(valid_metric[0])
    valid_loss_record.append(valid_loss)
    print(valid_acc_record,)
    return valid_metric, valid_loss, valid_repres_list, valid_label_list


def train_ACP1(train_iter, valid_iter, test_iter, model, optimizer, criterion, config, iter_k):
    steps = 0
    best_acc = 0
    best_performance = 0
    device = torch.device("cuda" if config.cuda else "cpu")

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.75, verbose=True)

    for epoch in range(1, config.epoch + 1):
        repres_list = []
        label_list = []
        counter = 0
        for batch in train_iter:

            input, label , seq = batch
            input,label = input.to(device),label.long().to(device)
            for i in range(len(seq)):
                if i == 0:
                    vec = torch.tensor(seq2vec[seq[0]]).to(device)
                else:
                    vec = torch.cat((vec, torch.tensor(seq2vec[seq[i]]).to(device)), dim=0)

            output = model.forward(input,vec)
            """VAE loss"""
            # logits,recon_x ,p_x, q_z = model.get_logits(input,vec)
            # loss2 = loss_function_original(recon_x, vec, p_x, q_z, criterion2)

            logits= model.get_logits(input, vec)
            loss1 = get_loss(logits,label,criterion )
            # loss = loss1+loss2
            loss = loss1

            counter = counter + 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

            '''Periodic Train Log'''
            if steps % config.interval_log == 0:
                corrects = (torch.max(logits, 1)[1] == label).sum()
                the_batch_size = label.shape[0]
                train_acc = 100.0 * corrects / the_batch_size
                sys.stdout.write(
                    '\rEpoch[{}] Batch[{}] - loss: {:.6f} | ACC: {:.4f}%({}/{})'.format(epoch, steps,
                                                                                        loss,
                                                                                        train_acc,
                                                                                        corrects,
                                                                                        the_batch_size))
                print()

                step_log_interval.append(steps)
                train_acc_record.append(train_acc)
                train_loss_record.append(loss)

        sum_epoch = iter_k * config.epoch + epoch
        lr_scheduler.step(loss)
        '''Periodic Validation'''
        if valid_iter and sum_epoch % config.interval_valid == 0:
            valid_metric, valid_loss, valid_repres_list, valid_label_list = periodic_valid(valid_iter,
                                                                                           model,
                                                                                           criterion,
                                                                                           config,
                                                                                           sum_epoch)
            valid_acc = valid_metric[0]
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_performance = valid_metric

        # '''Periodic Test'''
        if (test_iter and sum_epoch % config.interval_test == 0) :
            time_test_start = time.time()

            test_metric, test_loss, test_repres_list, test_label_list = periodic_test(test_iter,
                                                                                      model,
                                                                                      criterion,
                                                                                      config,
                                                                                      sum_epoch)
            '''Periodic Save'''
            # save the model if specific conditions are met
            test_acc = test_metric[0]

            if test_acc >= best_acc:
                best_acc = test_acc
                best_performance = test_metric
                if config.save_best and best_acc > config.threshold:
                    torch.save({"best_auc": best_acc, "model": model.state_dict()}, f'{best_acc}.pl')

            # test_label_list = [x + 2 for x in test_label_list]
            repres_list.extend(test_repres_list)
            label_list.extend(test_label_list)

            '''feature dimension reduction'''
            # print('epoch：',sum_epoch)
            # if sum_epoch == 1 or sum_epoch % 10 == 0 :
            #     dimension_reduction(repres_list, label_list, epoch,config)
            #
            # '''reduction feature visualization'''
            # if sum_epoch == 1 or sum_epoch % 10 == 0 or (epoch % 2 == 0 and epoch <= 10):
            #     penultimate_feature_visulization(repres_list, label_list, epoch,config)

    return best_performance


def model_eval(data_iter, model, criterion, config):
    device = torch.device("cuda" if config.cuda else "cpu")
    # device = 'cpu'
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)

    print('model_eval data_iter', len(data_iter))

    iter_size, corrects, avg_loss = 0, 0, 0
    repres_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for batch in data_iter:
            input, label,seq = batch
            label = torch.tensor(label, dtype=torch.long).to(device)
            input = input.to(device)
            lll = label.clone()
            label = torch.unsqueeze(label, 0)
            for i in range(len(seq)):
                if i == 0:
                    vec = torch.tensor(seq2vec[seq[0]]).to(device)
                else:
                    vec = torch.cat((vec, torch.tensor(seq2vec[seq[i]]).to(device)), dim=0)
            output = model.forward(input,vec)

            """VAE is used to process the embedded vectors of the pretrained model"""
            # logits, recon_x, p_x, q_z = model.get_logits(input, vec)
            # loss1 = loss_function_original(recon_x, vec, p_x, q_z, criterion2)# criterion2：nn.MSELOSS


            logits= model.get_logits(input, vec)
            repres_list.extend(output.cpu().detach().numpy())
            label_list.extend(lll.cpu().detach().numpy())

            logits = logits.view(-1, logits.size(-1))
            label = label.view(-1)
            label = label[1:-1]
            logits = logits[1:-1]


            loss2 = criterion(logits, label)
            # loss = loss1 + loss2
            loss = loss2
            # loss = (loss.float()).mean()
            avg_loss += loss.item()

            logits = torch.unsqueeze(logits, 0)
            label = torch.unsqueeze(label, 0)
            pred_prob_all = F.softmax(logits, dim=2)
            # Prediction probability [batch_size, seq_len, class_num]
            pred_prob_positive = pred_prob_all[:,:, 1]
            positive = torch.empty([0], device=device)
            # Probability of predicting positive classes [batch_size, seq_len]
            pred_prob_sort = torch.max(pred_prob_all, 2)
            # The maximum probability of prediction in each sample [batch_size]
            pred_class = pred_prob_sort[1]
            p_class = torch.empty([0], device=device)
            la = torch.empty([0], device=device)
            positive = torch.cat([positive, pred_prob_positive[0][:]])
            p_class = torch.cat([p_class, pred_class[0][:]])
            la = torch.cat([la, label[0][:]])

            corre = (pred_class == label).int()
            corrects += corre.sum()
            iter_size += label.size(1)
            label_pred = torch.cat([label_pred, p_class.float()])
            label_real = torch.cat([label_real, la.float()])
            pred_prob = torch.cat([pred_prob, positive])

    metric, roc_data, prc_data = util_metric.caculate_metric(label_pred, label_real, pred_prob)
    avg_loss /= len(data_iter)
    # accuracy = 100.0 * corrects / iter_size
    accuracy = metric[0]
    print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(avg_loss,
                                                                  100*accuracy,
                                                                  corrects,
                                                                  iter_size))

    return metric, avg_loss, repres_list, label_list, roc_data, prc_data



def k_fold_CV(train_iter_orgin, test_iter, config,):
    valid_performance_list = []

    fig1 = plt.figure(figsize=[12, 12])

    TPR = []
    meanFPR = np.linspace(0, 1, 100)
    i = 1
    for iter_k in range(config.k_fold):
        print('=' * 50, 'iter_k={}'.format(iter_k + 1), '=' * 50)

        # Cross validation on training set
        train_iter = [x for i, x in enumerate(train_iter_orgin) if i % config.k_fold != iter_k]
        valid_iter = [x for i, x in enumerate(train_iter_orgin) if i % config.k_fold == iter_k]
        print('----------Data Selection----------')
        print('train_iter index', [i for i, x in enumerate(train_iter_orgin) if i % config.k_fold != iter_k])
        print('valid_iter index', [i for i, x in enumerate(train_iter_orgin) if i % config.k_fold == iter_k])

        print('len(train_iter_orgin)', len(train_iter_orgin))
        print('len(train_iter)', len(train_iter))
        print('len(valid_iter)', len(valid_iter))
        if test_iter:
            print('len(test_iter)', len(test_iter))
        print('----------Data Selection Over----------')

        model = rnn.newModel()


        if config.cuda: model.cuda()
        # adjust_model(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.reg)
        criterion = nn.CrossEntropyLoss()
        # criterion2 = nn.MSELoss().to('cpu')
        model.train()

        print('=' * 50 + 'Start Training' + '=' * 50)
        valid_performance = train_ACP1(train_iter, valid_iter, test_iter, model, optimizer, criterion,config, iter_k)
        print('=' * 50 + 'Train Finished' + '=' * 50)

        print('=' * 40 + 'Cross Validation iter_k={}'.format(iter_k + 1), '=' * 40)
        valid_metric, valid_loss, valid_repres_list, valid_label_list, \
        valid_roc_data, valid_prc_data = model_eval(valid_iter, model, criterion, config)
        print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC,\tAP]')
        print(valid_metric.numpy())
        print('=' * 40 + 'Cross Validation Over' + '=' * 40)

        valid_performance_list.append(valid_performance)

        '''draw figure'''
        # draw_figure_CV(config, config.learn_name + '_k[{}]'.format(iter_k + 1))

        '''draw k_fold roc figure'''
        # fpr, tpr, rocauc =  valid_roc_data
        # TPR.append(interp(meanFPR, fpr, tpr))
        # plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, rocauc))
        # i= i+1

        '''reset plot data'''
        step_log_interval = []
        train_acc_record = []
        train_loss_record = []
        step_valid_interval = []
        valid_acc_record = []
        valid_loss_record = []

    return model, valid_performance_list

class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def train_test(train_iter, test_iter, config):

    """model construct"""
    model = rnn.newModel()

    # model.cuda()
    model.to('cpu')


    """pretrianed model fine-tunning"""
    # path = '../result/pre_cnn_attn/pre_cnn_num4layer8_lr.pt'
    # save_model = torch.load(path,map_location=torch.device('cpu'))
    # model_dict = model.state_dict()
    # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    # print(state_dict.keys())
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)
    # adjust_model(model)


    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=config.reg)
    criterion = nn.CrossEntropyLoss()

    print('=' * 50 + 'Start Training' + '=' * 50)
    best_performance = train_ACP1(train_iter, None, test_iter, model, optimizer,criterion,config, 0)
    print('=' * 50 + 'Train Finished' + '=' * 50)

    print('*' * 60 + 'The Last Test' + '*' * 60)
    last_test_metric, last_test_loss, last_test_repres_list, last_test_label_list, \
    last_test_roc_data, last_test_prc_data = model_eval(test_iter, model, criterion,config)
    print('[ACC,\t\tPrecision,\t\tSensitivity,\t\tSpecificity,\t\tF1,\t\tAUC,\t\tMCC,\t\tAP,\t\tTP,\t\tFP,\t\tTN,\t\tFN]')
    lmt = last_test_metric.numpy()
    print('%.5g\t\t' % lmt[0] , '%.5g\t\t' % lmt[1], '%.5g\t\t' % lmt[2], '%.5g\t\t' % lmt[3], '%.5g\t' % lmt[4], '%.5g\t\t' % lmt[5], '%.5g\t\t' % lmt[6],
          '%.5g\t\t' % lmt[7], '  %.5g\t\t' % lmt[8], '  %.5g\t\t' % lmt[9], ' %.5g\t\t' % lmt[10],' %.5g\t\t' % lmt[11])
    print('*' * 60 + 'The Last Test Over' + '*' * 60)
    result =[last_test_metric,last_test_roc_data,last_test_prc_data]
    # np.save(r'../dataset/another_data/padding/fusion/add.npy',result)
    return model, best_performance, last_test_metric

def get_w2c_data(train,test,emb,is_FL):

    if is_FL:
        x_test, test_label = get_data.load_tsv_format_data(test)  # 164
        x_train, train_label = get_data.load_tsv_format_data(train)  # 500
        data_x = x_train+x_test

        data_all, seq_all = get_data.genData(data_x)
        train_data, train_seq = data_all[:844],seq_all[:844] #the number of sequence on train set
        test_data, test_seq = data_all[844:], seq_all[844:] #the number of sequence on test set

        one_data = get_feature.get_one(data_x, is_one=True)
        train_one_data = one_data[:844,:,:]
        test_one_data = one_data[844:, :, :]
        print('train_one_data:', train_one_data.shape)
        print('test_one_data:', test_one_data.shape)

        pp_data = get_feature.get_one(data_x, is_one=False)
        train_pp_data = pp_data[:844, :, :]
        test_pp_data = pp_data[844:, :, :]

        print('train_pp_data:', train_pp_data.shape)
        print('test_pp_data:', test_pp_data.shape)
    else:
        x_test, test_label = get_data.get_sequence(test)  # 342
        x_train, train_label = get_data.get_sequence(train)  # 1376
        train_data,train_seq = get_data.genData(x_train)
        test_data,test_seq = get_data.genData(x_test)

        print("train sequence:",train_data.shape)
        print("test sequence:", test_data.shape)

        #max_len = 50/97/145/207
        #padding = NT CT
        #is_one:True->one-hot/False->PCPF
        # train_one_data = get_feature.get_one_new(x_train,is_one=True,max_len=207,padding='CT')
        # test_one_data = get_feature.get_one_new(x_test,is_one=True,max_len=207,padding='CT')
        train_one_data = get_feature.get_one(x_train, is_one=True)
        test_one_data = get_feature.get_one(x_test, is_one=True)
        print('train_one_data:',train_one_data.shape)
        print('test_one_data:',test_one_data.shape)

        # train_pp_data = get_feature.get_one_new(x_train,is_one=False,max_len=207,padding='CT')
        # test_pp_data = get_feature.get_one_new(x_test,is_one=False,max_len=207,padding='CT')
        train_pp_data = get_feature.get_one(x_train, is_one=False)
        test_pp_data = get_feature.get_one(x_test, is_one=False)
        print('train_pp_data:',train_pp_data.shape)
        print('test_pp_data:',test_pp_data.shape)


    print('************pretrain model embeddung sequence**********')
    seq2vec = json.load(open(emb))


    hc_train = torch.cat([train_one_data,train_pp_data],dim=2)
    print('hc_train:', hc_train.shape)

    #ACP-mix80,need to padding,another dataset dont need to append zeros-tensor alone.
    # new_train = torch.zeros([478,1,25])
    hc_test = torch.cat([test_one_data,test_pp_data],dim=2)
    print('hc_test:', hc_test.shape)
    new_test = torch.zeros([122,124,25])
    hc_test_new = torch.cat([hc_test,new_test],dim=1)
    print('hc_test_new:', hc_test_new.shape)

    class MyDataSet(Data.Dataset):
        def __init__(self, data, label, seq):
            self.data = data
            self.label = label
            self.seq = seq

        def __len__(self):
            return len(self.label)

        def __getitem__(self, idx):
            return self.data[idx], self.label[idx], self.seq[idx]

    train_dataset = MyDataSet(hc_train,train_label, train_seq)
    test_dataset = MyDataSet(hc_test_new, test_label, test_seq)

    batch_size = 64
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_iter,test_iter,seq2vec

def load_config():
    '''The following variables need to be actively determined for each training session:
       1.train-name: Name of the training
       2.path-config-data: The path of the model configuration. 'None' indicates that the default configuration is loaded
       3.path-train-data: The path of training set
       4.path-test-data: Path to test set

       Each training corresponds to a result folder named after train-name, which contains:
       1.report: Training report
       2.figure: Training figure
       3.config: model configuration
       4.model_save: model parameters
       5.others: other data
       '''

    '''Set the required variables in the configuration'''
    train_name = 'ACP'
    path_config_data = None
    # path_train_data, path_test_data = select_dataset()

    '''Get configuration'''
    if path_config_data is None:
        config = cf.get_train_config()
    else:
        config = pickle.load(open(path_config_data, 'rb'))

    '''Modify default configuration'''
    # config.epoch = 50

    '''Set other variables'''
    # flooding method
    b = 0.06

    config.if_multi_scaled = False

    '''initialize result folder'''
    result_folder = '../result/' + config.learn_name
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    '''Save all variables in configuration'''
    config.train_name = train_name
    # config.path_train_data = path_train_data
    # config.path_test_data = path_test_data

    config.b = b
    # config.if_multi_scaled = if_multi_scaled
    # config.model_name = model_name
    config.result_folder = result_folder

    return config

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    np.set_printoptions(linewidth=400, precision=4)
    time_start = time.time()
    seed_torch()
    '''load configuration'''
    config = load_config()

    '''set device'''
    # torch.cuda.set_device(config.device)
    torch.device('cpu')

    '''load data'''
    train_dir = 'ACP-Mixed80_train.fasta'
    test_dir = 'ACP-Mixed80_test.fasta'
    emd_dir = '../data/pretrain/protbert_ACP_mixed80.emb'

    """If dataset is in csv format, FL is True; If fasta, FL is false."""
    train_iter, test_iter,seq2vec = get_w2c_data(train_dir, test_dir, emd_dir,is_FL=False)
    # print(seq2vec)
    print('len_train_iter:',len(train_iter))
    # print('len_valid_iter:', len(valid_iter))
    print('len_test_iter:',len(test_iter))
    print('=' * 20, 'load data over', '=' * 20)

    '''draw preparation'''
    step_log_interval = []
    train_acc_record = []
    train_loss_record = []
    step_valid_interval = []
    valid_acc_record = []
    valid_loss_record = []
    valid_mcc_record = []
    step_test_interval = []
    test_acc_record = []
    test_loss_record = []
    test_mcc_record = []

    '''train procedure'''
    valid_performance = 0
    best_performance = 0
    last_test_metric = 0

    if config.k_fold == -1:
        # train and test
        model, best_performance, last_test_metric = train_test(train_iter, test_iter, config)
    else:
        # k cross validation
        model, test_performance_list = k_fold_CV(train_iter, test_iter, config)  # none

    '''draw figure'''
    draw_figure_train_test(config, config.learn_name)

    '''report result'''
    print('*=' * 50 + 'Result Report' + '*=' * 50)
    if config.k_fold != -1:
        print('test_performance_list', test_performance_list)
        tensor_list = [x.view(1, -1) for x in test_performance_list]
        cat_tensor = torch.cat(tensor_list, dim=0)
        metric_mean = torch.mean(cat_tensor, dim=0)

        print('valid mean performance')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC,\tAP,\ttp,\tfp,\ttn,\tfn]')
        print('\t{}'.format(metric_mean.numpy()))

        print('valid_performance list')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC,\tAP,\ttp,\tfp,\ttn,\tfn]')
        for tensor_metric in test_performance_list:
            print('\t{}'.format(tensor_metric.numpy()))
    else:
        print('last test performance')
        plmt = last_test_metric.numpy()
        print('%.5g\t\t' % plmt[0], '%.5g\t\t' % plmt[1], '%.5g\t\t' % plmt[2], '%.5g\t\t' % plmt[3],
              '%.5g\t' % plmt[4],'%.5g\t\t' % plmt[5], '%.5g\t\t' % plmt[6], '%.5g\t\t' % plmt[7],
              '%.5g\t\t' % plmt[8],'%.5g\t\t' % plmt[9], '%.5g\t\t' % plmt[10],'%.5g\t\t' % plmt[11])

        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC,\tAP]')
        print('\t{}'.format(last_test_metric))
        print()
        print('best_performance')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC,\tAP]')
        print('\t{}'.format(best_performance))

    '''save train result'''
    # save the model if specific conditions are met
    if config.k_fold == -1:
        best_acc = best_performance[0]
        last_test_acc = last_test_metric[0]
        if last_test_acc >= best_acc:
            best_acc = last_test_acc
            best_performance = last_test_metric
            if config.save_best and best_acc >= config.threshold:
                save_model(model.state_dict(), best_acc, config.result_folder, config.learn_name)

    # save the model configuration
    with open(config.result_folder + '/config.pkl', 'wb') as file:
        pickle.dump(config, file)
    print('-' * 50, 'Config Save Over', '-' * 50)

    time_end = time.time()
    print('total time:',time_end)