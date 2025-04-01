#!/usr/bin/env python
# coding=gbk

import argparse
import os
import random
import time
import warnings
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import shutil
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.StartUp import StartUp
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, average_precision_score


# 线性分类器数据集类
class LinearClassifierDataset(Dataset):
    def __init__(self, data, gpn_msa, hyena_dna, train=True):
        self.data = data
        self.gpn_msa = gpn_msa
        self.hyena_dna = hyena_dna
        self.train = train

        # 假设data中包含标签列，命名为'label'
        self.labels = torch.tensor(data['Label'].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gpn_feature = self.gpn_msa[idx].clone().detach()
        hyena_feature = self.hyena_dna[idx].clone().detach()
        label = self.labels[idx]

        return (gpn_feature, hyena_feature), label


class Classifier(nn.Module):
    def __init__(self, input_length, output_dim):
        super(Classifier, self).__init__()

        # 定义与 fc5 相似的网络结构
        self.fc5 = nn.Sequential(
            nn.Linear(input_length, 256),  # 从输入维度到256
            nn.BatchNorm1d(256),
            nn.Mish(),  # 激活函数 Mish
            nn.Linear(256, 64),  # 从256到64
            nn.BatchNorm1d(64),
            nn.Mish(),  # 激活函数 Mish
            nn.Linear(64, output_dim)  # 最后一层输出维度为 output_dim
        )

        # 对fc5中的线性层进行自定义的初始化
        self._initialize_weights()

        # 可选的批归一化，如果需要的话
        self.bn = None  # 如果需要的话可以将它替换为nn.BatchNorm1d等

    def forward(self, x):
        # 直接通过fc5网络进行前向传播
        x = self.fc5(x)
        return x

    def _initialize_weights(self):
        # 对fc5中的每一层进行初始化
        for m in self.fc5:
            if isinstance(m, nn.Linear):  # 只对Linear层进行初始化
                m.weight.data.normal_(mean=0.0, std=0.01)  # 使用正态分布初始化权重
                m.bias.data.zero_()  # 将偏置初始化为零


def select_random_samples(args, data, gpn_msa, hyena_dna, sample_ratio=0.1):
    """随机选择正样本和负样本"""
    positive_samples = data[data['Label'] == 1]
    negative_samples = data[data['Label'] == 0]

    # 计算需要选择的样本数量
    num_positive = int(len(positive_samples) * sample_ratio)
    num_negative = int(len(negative_samples) * sample_ratio)

    # 随机选择样本
    selected_positive = positive_samples.sample(n=num_positive, random_state=args.seed)
    selected_negative = negative_samples.sample(n=num_negative, random_state=args.seed)

    # 合并选择的样本
    selected_data = pd.concat([selected_positive, selected_negative])
    selected_gpn_msa = gpn_msa[selected_data.index]
    selected_hyena_dna = hyena_dna[selected_data.index]

    return selected_data, selected_gpn_msa, selected_hyena_dna


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """简化的二分类准确率计算"""
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target)
        acc = correct.float().mean() * 100
        return [acc, acc]  # 返回的 acc 是一个标量张量


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # 添加时间戳
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f'checkpoint_{timestamp}.pth.tar'

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'model_best_{timestamp}.pth.tar')


def train(train_loader, model, classifier, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    classifier.train()

    end = time.time()
    for i, ((gpn, hyena), target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            gpn = gpn.cuda(args.gpu, non_blocking=True)
            hyena = hyena.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        
        #清空梯度
        optimizer.zero_grad()
        
        # compute output
        output = model(gpn, hyena)
        output = classifier(output)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step() #优化器更新

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), gpn.size(0))
        top1.update(acc1.item(), gpn.size(0))  # 使用 item() 方法获取标量值

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return epoch, losses.avg, top1.avg


def save_checkpoint(state, is_best, save_dir="./finetune", filename="finetune.pth.tar"):
    # 添加版本控制和定期清理
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, filename)

    # 添加错误处理
    try:
        torch.save(state, filepath)
        if is_best:
            best_filepath = os.path.join(save_dir, 'finetune_model_best.pth.tar')
            shutil.copyfile(filepath, best_filepath)
            print(f"Saved best model to {best_filepath}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def cross_validate(train_dataset, args):
    # 创建5折交叉验证的分割器
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    metrics_results = {
        'SEN': [], 'SPE': [], 'PRE': [], 'F1': [],
        'MCC': [], 'ACC': [], 'AUC': [], 'AUPR': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f"\nTraining Fold {fold + 1}/5")

        model = StartUp(
            gpn_embedding_dim=args.gpn_dim,
            hyena_embedding_dim=args.hyena_dim,
            n_filters=args.n_filters,
            filter_sizes=args.filter_sizes,
            output_dim=args.num_classes,
            dropout=args.dropout
        )
        classifier = Classifier(input_length=args.input_dim, output_dim=args.num_classes)

        # model to gpu
        model = model.cuda(args.gpu)
        classifier = classifier.cuda(args.gpu)

        # 加载预训练模型
        if args.pretrained:
            if os.path.isfile(args.pretrained):
                print(f"=> loading checkpoint '{args.pretrained}'")
                checkpoint = torch.load(args.pretrained, map_location="cpu")

                # 提取预训练的 state_dict
                state_dict = checkpoint['state_dict']
                #print("State dict keys:", state_dict.keys())  # 输出所有键

                # 重新命名和过滤键
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("fea_encoder"):
                        new_key = k[len("fea_encoder."):]  # 移除 "fea_encoder." 前缀
                        new_state_dict[new_key] = v

                # 加载到模型中
                msg = model.load_state_dict(new_state_dict, strict=False)
                print(f"=> loaded pre-trained model '{args.pretrained}'")
            else:
                print(f"=> no checkpoint found at '{args.pretrained}'")

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = torch.optim.SGD(list(model.parameters()) + list(classifier.parameters()), args.learning_rate,
                                    weight_decay=args.weight_decay)
        cudnn.benchmark = True

        # 创建当前折的数据加载器
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        fold_train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_subsampler,
            num_workers=args.workers, pin_memory=True)
        fold_val_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=val_subsampler,
            num_workers=args.workers, pin_memory=True)
        
        best_metrics = None
        best_acc1 = 0
        # 训练当前折
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch, args)
            
            # 训练一个epoch
            _, _, acc1 = train(fold_train_loader, model, classifier, criterion, optimizer, epoch, args)
            
            # 在验证集上评估
            acc, probs, preds, current_metrics = validate(fold_val_loader, model, classifier, criterion, args)
        
        # 保存当前折的最佳结果
        print(f"\nFold {fold + 1} Results:")
        for metric, value in current_metrics.items():
            metrics_results[metric].append(value)
            print(f"{metric}: {value:.4f}")

        #清理内存
        del model, classifier, optimizer
        torch.cuda.empty_cache()
        
    # 计算并打印最终的平均结果
    print(f"\n{'='*20} Final five-fold cross validation results {'='*20}")
    for metric in metrics_results.keys():
        values = metrics_results[metric]
        mean_value = np.mean(values)
        std_value = np.std(values)
        print(f"{metric}: {mean_value:.4f} +- {std_value:.4f}")
    
    return metrics_results


def validate(val_loader, model, classifier, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # 初始化混淆矩阵的组件
    all_targets = []
    all_preds = []
    all_probs = []

    # 切换到评估模式
    model.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for i, ((gpn, hyena), target) in enumerate(val_loader):
            if args.gpu is not None:
                gpn = gpn.cuda(args.gpu, non_blocking=True)
                hyena = hyena.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # 计算输出
            fea = model(gpn, hyena)
            output = classifier(fea)

            # 获取预测概率和标签
            probs = torch.softmax(output, dim=1)[:, 1]
            preds = (probs > 0.5).long()

            # 检查 NaN 值并处理
            if torch.isnan(probs).any() or torch.isnan(preds).any():
                print(f"Warning: NaN values found in batch {i}")
                continue

            # 收集所有预测和目标
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # 测量时间
            batch_time.update(time.time() - end)
            end = time.time()

    # 转换为numpy数组
    all_targets = np.array(all_targets)  # 真实标签
    all_preds = np.array(all_preds)  # 预测标签
    all_probs = np.array(all_probs)  # 预测概率

    # 添加检查以确保有有效的预测和目标
    if len(all_targets) == 0 or len(all_preds) == 0:
        print("Error: No valid predictions")
        return 0, [], [], {}

    # 计算各项指标
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()

    metrics = {
        'SEN': tp / (tp + fn),
        'SPE': tn / (tn + fp),
        'PRE': precision_score(all_targets, all_preds),
        'F1': f1_score(all_targets, all_preds),
        'MCC': matthews_corrcoef(all_targets, all_preds),
        'ACC': (tp + tn) / (tp + tn + fp + fn),
        'AUC': roc_auc_score(all_targets, all_probs),
        'AUPR': average_precision_score(all_targets, all_probs)
    }

    # 打印当前评估指标
    for metric, value in metrics.items():
        print(f' * {metric}: {value:.4f}')

    return metrics['ACC'], all_probs, all_preds, metrics


def predict(val_loader, model, classifier, criterion, args):
    # 切换到评估模式
    model.eval()
    classifier.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        end = time.time()
        for i, ((gpn, hyena), target) in enumerate(val_loader):
            if args.gpu is not None:
                gpn = gpn.cuda(args.gpu, non_blocking=True)
                hyena = hyena.cuda(args.gpu, non_blocking=True)

            # 计算输出
            fea = model(gpn, hyena)
            output = classifier(fea)

            # 获取预测概率和标签
            probs = torch.softmax(output, dim=1)[:, 1]
            preds = (probs > 0.5).long()

            # 检查 NaN 值并处理
            if torch.isnan(probs).any() or torch.isnan(preds).any():
                print(f"Warning: NaN values found in probabilities or predictions at batch {i}.")
                continue  # 跳过当前批次

            # 收集所有预测和目标
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 转换为numpy数组
    all_preds = np.array(all_preds)  # 预测标签
    all_probs = np.array(all_probs)  # 预测概率

    return all_probs, all_preds  # 仍然返回准确率作为主要指标


def main():
    parser = argparse.ArgumentParser(description='Linear Classification')
    parser.add_argument('--data', metavar="DIR", default="./dataset", help="path to dataset")
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('-b', '--batch_size', default=256, type=int, help='mini-batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.5, type=float, help='initial learning rate')
    parser.add_argument('--schedule', default=[10, 25], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--gpn_dim', default=768, type=int, help='GPN embedding dimension')
    parser.add_argument('--hyena_dim', default=128, type=int, help='HyenaDNA embedding dimension')
    parser.add_argument('--input_dim', default=1024, type=int, help='input dimension of classifier')
    parser.add_argument('--n_filters', default=64, type=int, help='number of filters')
    parser.add_argument('--filter_sizes', default=[3, 4, 5], type=int, nargs='+', help='filter sizes')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('-wd', '--weight_decay', default=0., type=float, help='weight decay')
    parser.add_argument('-p', '--print_freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model')
    parser.add_argument('--seed', default=666, type=int, help='seed for initializing training')
    parser.add_argument("--model_save_dir", default="./finetune", type=str, help="path to save finetune model")
    parser.add_argument('--gpu', default=1, type=int, help='GPU id to use')
    parser.add_argument('--cross_validation', action='store_true', help='whether to perform 5-fold cross validation')
    parser.add_argument('--high_confi', action='store_true', help='whether to train by 990 sample')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU.')
        torch.cuda.set_device(args.gpu)

    # 加载数据
    try:
        # 加载训练集
        if args.high_confi:
            print("\n=> Starting training with 990 samples")
            train_data = pd.read_csv(os.path.join(args.data, "Training_990pos_990neg_start-lost_hg38_1001bp_seq.txt"), sep="\t", header=0)
            print("训练标签加载成功，数据形状:", train_data.shape)
            train_gpn_msa = torch.load(os.path.join(args.data, "Training_990pos_990neg_start-lost_hg38_GPN-MSA_alt_feature.pth"))
            print("训练GPN-MSA特征加载成功,数据形状:", train_gpn_msa.shape)
            train_hyena_dna = torch.load(os.path.join(args.data, "Training_990pos_990neg_start-lost_hg38_1001bp_HyenaDNA_feature.pth"))
            print("训练HyenaDNA特征加载成功,数据形状:", train_hyena_dna.shape)
        
        else:
            print("\n=> Starting training with 1264 samples")
            train_data = pd.read_csv(os.path.join(args.data, "1-Training_Label_1264pos_1264neg_start-lost_hg38_1001bp_kmer_random.csv"), sep=",", header=0)
            print("训练标签加载成功，数据形状:", train_data.shape)
            train_gpn_msa = torch.load(os.path.join(args.data, "1-Training_Label_1264pos_1264neg_start-lost_hg38_GPN-MSA_alt_feature.pth"))
            print("训练GPN-MSA特征加载成功,数据形状:", train_gpn_msa.shape)
            train_hyena_dna = torch.load(os.path.join(args.data, "1-Training_Label_1264pos_1264neg_start-lost_hg38_1001bp_HyenaDNA_feature.pth"))
            print("训练HyenaDNA特征加载成功,数据形状:", train_hyena_dna.shape)

        # 加载测试集一
        val_data = pd.read_csv(os.path.join(args.data, "Testing_Label_700pos_486neg_start-lost_hg38_1001bp_kmer_random.txt"), sep="\t", header=0)
        print("测试集一标签加载成功，数据形状:", val_data.shape)
        val_gpn_msa = torch.load(os.path.join(args.data, "Testing_Label_700pos_486neg_start-lost_hg38_GPN-MSA_alt_feature.pth"))
        print("测试集一GPN-MSA特征加载成功,数据形状:", val_gpn_msa.shape)
        val_hyena_dna = torch.load(os.path.join(args.data, "Testing_Label_700pos_486neg_start-lost_hg38_1001bp_random_HyenaDNA_feature.pth"))
        print("测试集一HyenaDNA特征加载成功,数据形状:", val_hyena_dna.shape)

        # 加载测试集二
        test_data = pd.read_csv(os.path.join(args.data, "HGMD-ClinVar_new_startloss_1001bp_seq.txt"), sep="\t", header=0)
        print("测试集二标签加载成功，数据形状:", test_data.shape)
        test_gpn_msa = torch.load(os.path.join(args.data, "HGMD-ClinVar_new_startloss_GPN-MSA_alt_feature.pth"))
        print("测试集二GPN-MSA特征数据加载成功,数据形状:", test_gpn_msa.shape)
        test_hyena_dna = torch.load(os.path.join(args.data, "HGMD-ClinVar_new_startloss_1001bp_HyenaDNA_feature.pth"))
        print("测试集二HyenaDNA特征数据加载成功,数据形状:", test_hyena_dna.shape)

#        # 加载测试子集
#        notrain_data = pd.read_csv(os.path.join(args.data, "Testing1_subset_NoTrain_830data_1001bp_seq.txt"), sep="\t", header=0)
#        print("未训练子集标签加载成功，数据形状:", notrain_data.shape)
#        notrain_gpn_msa = torch.load(os.path.join(args.data, "Testing1_subset_NoTrain_830data_GPN-MSA_alt_feature.pth"))
#        print("未训练子集GPN-MSA特征数据加载成功,数据形状:", notrain_gpn_msa.shape)
#        notrain_hyena_dna = torch.load(os.path.join(args.data, "Testing1_subset_NoTrain_830data_1001bp_HyenaDNA_feature.pth"))
#        print("未训练子集HyenaDNA特征数据加载成功,数据形状:", notrain_hyena_dna.shape)

#        # 随机选择部分正样本和负样本进行模型微调
#        train_data, train_gpn_msa, train_hyena_dna = select_random_samples(args, train_data, train_gpn_msa, train_hyena_dna, sample_ratio=0.1)
#        print("随机选择的训练数据形状:", train_data.shape)

#        # 加载数据集
#        pre_data = pd.read_csv(os.path.join(args.data, "Human_start-lost_list_180350_hg38_1001bp_seq.txt"), sep="\t", header=0)
#        if 'Label' not in pre_data.columns:
#            pre_data['Label'] = -1  # 添加 'Label' 列并设置值为 -1
#        print("数据集标签加载成功,数据形状:", pre_data.shape)
#        pre_gpn_msa = torch.load(os.path.join(args.data, "Human_start-lost_180350data_GPN-MSA_alt_feature.pth"))
#        print("数据集GPN-MSA特征数据加载成功, 数据形状:", pre_gpn_msa.shape)
#        pre_hyena_dna = torch.load(os.path.join(args.data, "Human_start-lost_180350data_1001bp_HyenaDNA_feature.pth"))
#        print("数据集HyenaDNA特征数据加载成功,数据形状:", pre_hyena_dna.shape)

        # 创建数据集
        train_dataset = LinearClassifierDataset(
            train_data,
            train_gpn_msa,
            train_hyena_dna,
            train=True
        )

        val_dataset = LinearClassifierDataset(
            val_data,
            val_gpn_msa,
            val_hyena_dna,
            train=False
        )

        test_dataset = LinearClassifierDataset(
            test_data,
            test_gpn_msa,
            test_hyena_dna,
            train=False
        )

#        notrain_dataset = LinearClassifierDataset(
#            notrain_data,
#            notrain_gpn_msa,
#            notrain_hyena_dna,
#            train=False
#        )

#        pre_dataset = LinearClassifierDataset(
#            pre_data,
#            pre_gpn_msa,
#            pre_hyena_dna,
#            train=False
#        )

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

#    notrain_loader = DataLoader(
#        notrain_dataset,
#        batch_size=args.batch_size,
#        shuffle=False,
#        num_workers=args.workers,
#        pin_memory=True
#    )

#    pre_loader = DataLoader(
#        pre_dataset,
#        batch_size=args.batch_size,
#        shuffle=False,
#        num_workers=args.workers,
#        pin_memory=True
#    )

    # 进行交叉验证
    if args.cross_validation:
        print("\n=> Starting five-fold cross validation")
        # 进行5折交叉验证
        metrics_results = cross_validate(train_dataset, args)

    else:
        print("\n=> Starting training with whole training set")
        # 创建模型
        model = StartUp(
            gpn_embedding_dim=args.gpn_dim,
            hyena_embedding_dim=args.hyena_dim,
            n_filters=args.n_filters,
            filter_sizes=args.filter_sizes,
            output_dim=args.num_classes,
            dropout=args.dropout
        )
        classifier = Classifier(input_length=args.input_dim, output_dim=args.num_classes)

        # model to gpu
        model = model.cuda(args.gpu)
        classifier = classifier.cuda(args.gpu)

        # 加载预训练模型
        if args.pretrained:
            if os.path.isfile(args.pretrained):
                print(f"=> loading checkpoint '{args.pretrained}'")
                checkpoint = torch.load(args.pretrained, map_location="cpu")

                # 提取预训练的 state_dict
                state_dict = checkpoint['state_dict']
                print("State dict keys:", state_dict.keys())  # 输出所有键

                # 重新命名和过滤键
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("fea_encoder"):
                        new_key = k[len("fea_encoder."):]  # 移除 "fea_encoder." 前缀
                        new_state_dict[new_key] = v

                # 加载到模型中
                msg = model.load_state_dict(new_state_dict, strict=False)
                print(f"=> loaded pre-trained model '{args.pretrained}'")
            else:
                print(f"=> no checkpoint found at '{args.pretrained}'")

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = torch.optim.SGD(list(model.parameters()) + list(classifier.parameters()), args.learning_rate,
                                    weight_decay=args.weight_decay)
        cudnn.benchmark = True

        # 可选：从检查点恢复
        best_acc1 = 0
        to_restore = {"epoch": 0}
        start_epoch = to_restore["epoch"]

        # 训练循环
        for epoch in range(start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args)

            # 训练一个epoch
            _, _, acc1 = train(train_loader, model, classifier, criterion, optimizer, epoch, args)

            # 在测试集一上评估
            print("\n在测试集一上评估:")
            validate(val_loader, model, classifier, criterion, args)

            # 记录最佳acc并保存检查点
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                is_best=False,
                save_dir=args.model_save_dir,
                filename=f"finetune_{epoch:04d}.pth.tar"
            )


        # 在测试集一上进行最终评估
        # print("\n在测试集一上评估:")
        # validate(val_loader, model, classifier, criterion, args)

        # 在测试集二上进行最终评估
        print("\n在测试集二上评估:")
        validate(test_loader, model, classifier, criterion, args)

#        # 在未训练子集上进行最终评估
#        print("\n在未训练子集上评估:")
#        validate(notrain_loader, model, classifier, criterion, args)


#        # 在数据集上进行预测并输出
#        print("\n输出预测结果:")
#        pred_score, pred_label = predict(pre_loader, model, classifier, criterion, args)
#        pred_score_df = pd.DataFrame(pred_score, columns=['pred_score'])  # 将预测分数转换为 DataFrame
#        pred_label_df = pd.DataFrame(pred_label, columns=['pred_label'])  # 将预测标签转换为 DataFrame
#        merged_data = pd.concat([pre_data.iloc[:, :8], pred_score_df, pred_label_df], axis=1)
#        output_path = "./Human_start-lost_180350data_startCLR_pred_result.txt"
#        merged_data.to_csv(output_path, sep="\t", index=False)
#        print(f"数据集预测结果已保存至: {output_path}")

if __name__ == '__main__':
    main()
