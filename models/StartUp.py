#!/usr/bin/env python
#coding=gbk

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TextCNN_block1(nn.Module):
    def __init__(self, gpn_embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super(TextCNN_block1, self).__init__()
        self.convs1 = nn.ModuleList([nn.Conv1d(in_channels=gpn_embedding_dim,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])  # ���֦���
        self.fc1 = nn.Linear(1920, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.Mish1 = nn.Mish()
        self.batchnorm1 = nn.BatchNorm1d(512)


    def forward(self, gpn_fea, hyena_fea):
        # ���������ݽ���ά�ȱ任
        # ���������ݵ�ά�ȴ� [batch_size, sequence_length, embedding_dim] ��Ϊ [batch_size, embedding_dim, sequence_length]
        gpn_embedded = gpn_fea.permute(0, 2, 1)
        # Ӧ�þ���㲢�ϲ����
        gpn_conved = [self.Mish1(conv(gpn_embedded)) for conv in self.convs1] # nn.Conv1d ��Ҫ���� [batch_size, channels, sequence_length] ��״������

        # �ػ���
        gpn_pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in gpn_conved]

        # ���֧����չ��
        gpn_flatten = [pool.contiguous().view(pool.size(0), -1) for pool in gpn_pooled]

        # ������֧������һ��
        gpn_cat = self.dropout1(torch.cat(gpn_flatten, dim=1))  ##���֧���Ӻ�ά��Ϊ��n_filters * filter_sizes * 10
        gpn_cat_i = self.fc1(gpn_cat)   #ʹ�����Բ����ά�ȱ任
        gpn_cat_i = self.batchnorm1(gpn_cat_i)

        # �������������
        return gpn_cat_i


class TextCNN_block2(nn.Module):
    def __init__(self, hyena_embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super(TextCNN_block2, self).__init__()
        self.convs2 = nn.ModuleList([nn.Conv1d(in_channels=hyena_embedding_dim,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])  # ���֦���
        self.fc3 = nn.Linear(1920, 512)
        self.dropout2 = nn.Dropout(dropout)
        self.Mish2 = nn.Mish()
        self.batchnorm2 = nn.BatchNorm1d(512)


    def forward(self, gpn_fea, hyena_fea):
        # ���������ݽ���ά�ȱ任
        # ���������ݵ�ά�ȴ� [batch_size, sequence_length, embedding_dim] ��Ϊ [batch_size, embedding_dim, sequence_length]
        hyena_embedded = hyena_fea.permute(0, 2, 1)
        # Ӧ�þ���㲢�ϲ����
        hyena_conved = [self.Mish2(conv(hyena_embedded)) for conv in self.convs2] # nn.Conv1d ��Ҫ���� [batch_size, channels, sequence_length] ��״������

        # �ػ���
        hyena_pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in hyena_conved]

        # ���֧����չ��
        hyena_flatten = [pool.contiguous().view(pool.size(0), -1) for pool in hyena_pooled]

        # ������֧������һ��
        hyena_cat = self.dropout2(torch.cat(hyena_flatten, dim=1))  ##���֧���Ӻ�ά��Ϊ��n_filters * filter_sizes * 10
        hyena_cat_i = self.fc3(hyena_cat)   #ʹ�����Բ����ά�ȱ任
        hyena_cat_i = self.batchnorm2(hyena_cat_i)

        # �������������
        return hyena_cat_i
        

class StartUp(nn.Module):
    def __init__(self, gpn_embedding_dim, hyena_embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super(StartUp, self).__init__()

        self.gpn_encoder = TextCNN_block1(gpn_embedding_dim, n_filters, filter_sizes, output_dim, dropout)
        self.hyena_encoder = TextCNN_block2(hyena_embedding_dim, n_filters, filter_sizes, output_dim, dropout)      

    def forward(self, gpn_fea, hyena_fea):
        # ���������ݽ��д���
        data1 = self.gpn_encoder(gpn_fea, hyena_fea) #GPN-MSA��TextCNN������
        data2 = self.hyena_encoder(gpn_fea, hyena_fea) #ͻ�����м�����Ŵ����ε��ں�����

        # �������������ƴ��
        fea = torch.cat([data1, data2], dim=1)

        # �������������
        return fea