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


# ��ȡ����
test_data = pd.read_table("../dataset/sample_HyenaDNA_input.txt", header=None) ###

# select model
pretrained_model_name = 'hyenadna-tiny-1k-seqlen'  # use None if training from scratch
pretrained_model_path = './HuggingFace'

# �������г��ȣ�hyenadna-tiny-1k-seqlen�����1024bp��
max_lengths = {
    'hyenadna-tiny-1k-seqlen': 1024,
    'hyenadna-small-32k-seqlen': 32768,
    'hyenadna-medium-160k-seqlen': 160000,
    'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
    'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
}
max_length = max_lengths[pretrained_model_name]  # auto selects

batch_size = 4
use_padding = True #�����Ȳ����Ƿ�Ҫ���
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

# ��ʼ��һ��NumPy��������������
features_np = np.zeros((len(test_data), 1003, 128), dtype=np.float32) #�ڶ���ά�ȵ�������Ҫ�������г������޸ģ��������������г���+2

# ���������б���ȡ����
with torch.inference_mode(): #
    # ѭ��������������
    for i in range(len(test_data)):
        sequence = test_data.iloc[i,0]
        tok_seq = tokenizer(sequence)
        tok_seq = tok_seq["input_ids"]  # grab ids        
        # place on device, convert to tensor
        tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)  # unsqueeze for batch dim
        tok_seq = tok_seq.to(device = device)
        
        embeddings = model(tok_seq)
        
        # �������PyTorch����ת��ΪNumPy���鲢���浽features_np��
        features_np[i] = embeddings.cpu().numpy()  # ȷ�������CPU�ϣ�Ȼ��ת��ΪNumPy����


## ����������������ݣ���ʽ�����֣���ѡһ�ּ���
## 1.��NumPy���鱣��Ϊ.npy�ļ�
# features_tensor =  np.array(features_np)
# print(features_tensor.shape)
# np.save('/data4/yebin/hyena-dna/Liujie/test1_101bp_hynaDNA_feature.npy', features_tensor)   ###����ļ�

## 2.ֱ�ӱ���Ϊpytorch������ʽ��.pth�ļ�
fea_tensor = torch.from_numpy(features_np)
print(fea_tensor.shape)
torch.save(fea_tensor, '../dataset/sample_HyenaDNA_feature.pth')

