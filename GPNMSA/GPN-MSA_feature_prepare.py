#!/usr/bin/env python
#coding=gbk

from gpn.data import GenomeMSA, Tokenizer
import gpn.model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoModelForMaskedLM

# 设置模型路径和多序列比对（MSA）数据路径
model_path = './model'
msa_path = './89.zarr.zip'

# 读取数据
test_data = pd.read_csv("../dataset/sample_GPNMSA_input.csv", header=0)

# 加载MSA数据
genome_msa = GenomeMSA(msa_path)

# 加载模型
model = AutoModel.from_pretrained(model_path)
model.eval()

# 初始化一个NumPy数组来保存特征
features_np = np.zeros((len(test_data), 128, 768), dtype=np.float32)

# 遍历样本列表并提取特征
with torch.no_grad():  # 不需要计算梯度，节省内存和计算资源
    # 循环遍历测试数据
    for i in range(len(test_data)):
        # 从测试数据中获取相应的信息
        chrom = test_data.iloc[i, 0]  # 第一列
        start_pos = int(test_data.iloc[i, 1])  # 第二列的起始位置
        end_pos = int(test_data.iloc[i, 1])  # 第二列的结束位置

        # 获取标记的MSA数据
        msa = genome_msa.get_msa(chrom, start_pos-64, end_pos+64, strand="+", tokenize=True)

        # 将MSA数据转换为Tensor格式
        msa = torch.tensor(np.expand_dims(msa, 0).astype(np.int64))

        # 分离人类和其他物种的数据
        input_ids, aux_features = msa[:, :, 0], msa[:, :, 1:]

        # 嵌入表示
        embedding = model(input_ids=input_ids, aux_features=aux_features).last_hidden_state

        # 将输出从PyTorch张量转换为NumPy数组并保存到features_np中
        features_np[i] = embedding.cpu().numpy()  # 确保输出在CPU上，然后转换为NumPy数组

        # 嵌入可视化
#        embedding_df = pd.DataFrame(StandardScaler().fit_transform(embedding[0].numpy()))
#        embedding_df.index.name = "Position"
#        embedding_df.columns.name = "Embedding dimension"

# 将NumPy数组转换为PyTorch张量并保存为.pth文件
features_tensor = torch.from_numpy(features_np)
torch.save(features_tensor, '../dataset/sample_GPN-MSA_feature.pth')