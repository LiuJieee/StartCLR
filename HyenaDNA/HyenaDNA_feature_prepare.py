#!/usr/bin/env python
#coding=gbk

import json
import os
import subprocess
import numpy as np
import pandas as pd
# import transformers
from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
from standalone_hyenadna import HyenaDNAModel
from standalone_hyenadna import CharacterTokenizer
from load import inject_substring, load_weights, HyenaDNAPreTrainedModel

import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# 读取数据
test_data = pd.read_table("../dataset/sample_HyenaDNA_input.txt", header=None) ###

# select model
pretrained_model_name = 'hyenadna-tiny-1k-seqlen'  # use None if training from scratch
pretrained_model_path = './HuggingFace'

# 设置序列长度（hyenadna-tiny-1k-seqlen最长接受1024bp）
max_lengths = {
    'hyenadna-tiny-1k-seqlen': 1024,
    'hyenadna-small-32k-seqlen': 32768,
    'hyenadna-medium-160k-seqlen': 160000,
    'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
    'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
}
max_length = max_lengths[pretrained_model_name]  # auto selects

batch_size = 4
use_padding = True #若长度不够是否要填充
rc_aug = False  # reverse complement augmentation
add_eos = False  # add end of sentence token

# we need these for the decoder head, if using
use_head = False
n_classes = 2  # not used for embeddings only

# you can override with your own backbone config here if you want,
# otherwise we'll load the HF one in None
backbone_cfg = None

# use the pretrained Huggingface wrapper instead
model = HyenaDNAPreTrainedModel.from_pretrained(
    pretrained_model_path,
    pretrained_model_name,
    download=False,
    config=backbone_cfg,
    device=device,
    use_head=use_head,
    n_classes=n_classes,
)

model.to(device = device)
model.eval()

# create tokenizer
tokenizer = CharacterTokenizer(
    characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
    model_max_length=max_length + 2,  # to account for special tokens, like EOS
    add_special_tokens=False,  # we handle special tokens elsewhere
    padding_side='left', # since HyenaDNA is causal, we pad on the left
)

# 初始化一个NumPy数组来保存特征
features_np = np.zeros((len(test_data), 1003, 128), dtype=np.float32) #第二个维度的数字需要根据序列长度来修改：输入数据中序列长度+2

# 遍历样本列表并提取特征
with torch.inference_mode(): #
    # 循环遍历测试数据
    for i in range(len(test_data)):
        sequence = test_data.iloc[i,0]
        tok_seq = tokenizer(sequence)
        tok_seq = tok_seq["input_ids"]  # grab ids        
        # place on device, convert to tensor
        tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)  # unsqueeze for batch dim
        tok_seq = tok_seq.to(device = device)
        
        embeddings = model(tok_seq)
        
        # 将输出从PyTorch张量转换为NumPy数组并保存到features_np中
        features_np[i] = embeddings.cpu().numpy()  # 确保输出在CPU上，然后转换为NumPy数组


## 保存输出的特征数据，格式有两种，任选一种即可
## 1.将NumPy数组保存为.npy文件
# features_tensor =  np.array(features_np)
# print(features_tensor.shape)
# np.save('/data4/yebin/hyena-dna/Liujie/test1_101bp_hynaDNA_feature.npy', features_tensor)   ###输出文件

## 2.直接保存为pytorch张量形式的.pth文件
fea_tensor = torch.from_numpy(features_np)
print(fea_tensor.shape)
torch.save(fea_tensor, '../dataset/sample_HyenaDNA_feature.pth')

