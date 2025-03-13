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
                                    for fs in filter_sizes])  # 多分枝卷积
        self.fc1 = nn.Linear(1920, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.Mish1 = nn.Mish()
        self.batchnorm1 = nn.BatchNorm1d(512)


    def forward(self, gpn_fea, hyena_fea):
        # 对输入数据进行维度变换
        # 将输入数据的维度从 [batch_size, sequence_length, embedding_dim] 变为 [batch_size, embedding_dim, sequence_length]
        gpn_embedded = gpn_fea.permute(0, 2, 1)
        # 应用卷积层并合并结果
        gpn_conved = [self.Mish1(conv(gpn_embedded)) for conv in self.convs1] # nn.Conv1d 需要的是 [batch_size, channels, sequence_length] 形状的数据

        # 池化层
        gpn_pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in gpn_conved]

        # 多分支线性展开
        gpn_flatten = [pool.contiguous().view(pool.size(0), -1) for pool in gpn_pooled]

        # 将各分支连接在一起
        gpn_cat = self.dropout1(torch.cat(gpn_flatten, dim=1))  ##多分支连接后，维度为：n_filters * filter_sizes * 10
        gpn_cat_i = self.fc1(gpn_cat)   #使用线性层进行维度变换
        gpn_cat_i = self.batchnorm1(gpn_cat_i)

        # 输出特征并分类
        return gpn_cat_i


class TextCNN_block2(nn.Module):
    def __init__(self, hyena_embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super(TextCNN_block2, self).__init__()
        self.convs2 = nn.ModuleList([nn.Conv1d(in_channels=hyena_embedding_dim,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])  # 多分枝卷积
        self.fc3 = nn.Linear(1920, 512)
        self.dropout2 = nn.Dropout(dropout)
        self.Mish2 = nn.Mish()
        self.batchnorm2 = nn.BatchNorm1d(512)


    def forward(self, gpn_fea, hyena_fea):
        # 对输入数据进行维度变换
        # 将输入数据的维度从 [batch_size, sequence_length, embedding_dim] 变为 [batch_size, embedding_dim, sequence_length]
        hyena_embedded = hyena_fea.permute(0, 2, 1)
        # 应用卷积层并合并结果
        hyena_conved = [self.Mish2(conv(hyena_embedded)) for conv in self.convs2] # nn.Conv1d 需要的是 [batch_size, channels, sequence_length] 形状的数据

        # 池化层
        hyena_pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in hyena_conved]

        # 多分支线性展开
        hyena_flatten = [pool.contiguous().view(pool.size(0), -1) for pool in hyena_pooled]

        # 将各分支连接在一起
        hyena_cat = self.dropout2(torch.cat(hyena_flatten, dim=1))  ##多分支连接后，维度为：n_filters * filter_sizes * 10
        hyena_cat_i = self.fc3(hyena_cat)   #使用线性层进行维度变换
        hyena_cat_i = self.batchnorm2(hyena_cat_i)

        # 输出特征并分类
        return hyena_cat_i
        

class StartUp(nn.Module):
    def __init__(self, gpn_embedding_dim, hyena_embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super(StartUp, self).__init__()

        self.gpn_encoder = TextCNN_block1(gpn_embedding_dim, n_filters, filter_sizes, output_dim, dropout)
        self.hyena_encoder = TextCNN_block2(hyena_embedding_dim, n_filters, filter_sizes, output_dim, dropout)      

    def forward(self, gpn_fea, hyena_fea):
        # 对输入数据进行处理
        data1 = self.gpn_encoder(gpn_fea, hyena_fea) #GPN-MSA→TextCNN的特征
        data2 = self.hyena_encoder(gpn_fea, hyena_fea) #突变序列及表观遗传修饰的融合特征

        # 将两类输出特征拼接
        fea = torch.cat([data1, data2], dim=1)

        # 输出特征并分类
        return fea