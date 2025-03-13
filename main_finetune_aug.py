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


# ���Է��������ݼ���
class LinearClassifierDataset(Dataset):
    def __init__(self, data, gpn_msa, hyena_dna, train=True):
        self.data = data
        self.gpn_msa = gpn_msa
        self.hyena_dna = hyena_dna
        self.train = train

        # ����data�а�����ǩ�У�����Ϊ'label'
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

        # ������ fc5 ���Ƶ�����ṹ
        self.fc5 = nn.Sequential(
            nn.Linear(input_length, 256),  # ������ά�ȵ�256
            nn.BatchNorm1d(256),
            nn.Mish(),  # ����� Mish
            nn.Linear(256, 64),  # ��256��64
            nn.BatchNorm1d(64),
            nn.Mish(),  # ����� Mish
            nn.Linear(64, output_dim)  # ���һ�����ά��Ϊ output_dim
        )

        # ��fc5�е����Բ�����Զ���ĳ�ʼ��
        self._initialize_weights()

        # ��ѡ������һ���������Ҫ�Ļ�
        self.bn = None  # �����Ҫ�Ļ����Խ����滻Ϊnn.BatchNorm1d��

    def forward(self, x):
        # ֱ��ͨ��fc5�������ǰ�򴫲�
        x = self.fc5(x)
        return x

    def _initialize_weights(self):
        # ��fc5�е�ÿһ����г�ʼ��
        for m in self.fc5:
            if isinstance(m, nn.Linear):  # ֻ��Linear����г�ʼ��
                m.weight.data.normal_(mean=0.0, std=0.01)  # ʹ����̬�ֲ���ʼ��Ȩ��
                m.bias.data.zero_()  # ��ƫ�ó�ʼ��Ϊ��


def select_random_samples(args, data, gpn_msa, hyena_dna, sample_ratio=0.1):
    """���ѡ���������͸�����"""
    positive_samples = data[data['Label'] == 1]
    negative_samples = data[data['Label'] == 0]

    # ������Ҫѡ�����������
    num_positive = int(len(positive_samples) * sample_ratio)
    num_negative = int(len(negative_samples) * sample_ratio)

    # ���ѡ������
    selected_positive = positive_samples.sample(n=num_positive, random_state=args.seed)
    selected_negative = negative_samples.sample(n=num_negative, random_state=args.seed)

    # �ϲ�ѡ�������
    selected_data = pd.concat([selected_positive, selected_negative])
    selected_gpn_msa = gpn_msa[selected_data.index]
    selected_hyena_dna = hyena_dna[selected_data.index]

    return selected_data, selected_gpn_msa, selected_hyena_dna


class AverageMeter:
    """���㲢�洢ƽ��ֵ�͵�ǰֵ"""

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
    """�򻯵Ķ�����׼ȷ�ʼ���"""
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target)
        acc = correct.float().mean() * 100
        return [acc, acc]  # ���ص� acc ��һ����������


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # ���ʱ���
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
        
        #����ݶ�
        optimizer.zero_grad()
        
        # compute output
        output = model(gpn, hyena)
        output = classifier(output)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step() #�Ż�������

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), gpn.size(0))
        top1.update(acc1.item(), gpn.size(0))  # ʹ�� item() ������ȡ����ֵ

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return epoch, losses.avg, top1.avg


def save_checkpoint(state, is_best, save_dir="./finetune", filename="finetune.pth.tar"):
    # ��Ӱ汾���ƺͶ�������
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, filename)

    # ��Ӵ�����
    try:
        torch.save(state, filepath)
        if is_best:
            best_filepath = os.path.join(save_dir, 'finetune_model_best.pth.tar')
            shutil.copyfile(filepath, best_filepath)
            print(f"Saved best model to {best_filepath}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def cross_validate(train_dataset, args):
    # ����5�۽�����֤�ķָ���
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

        # ����Ԥѵ��ģ��
        if args.pretrained:
            if os.path.isfile(args.pretrained):
                print(f"=> loading checkpoint '{args.pretrained}'")
                checkpoint = torch.load(args.pretrained, map_location="cpu")

                # ��ȡԤѵ���� state_dict
                state_dict = checkpoint['state_dict']
                #print("State dict keys:", state_dict.keys())  # ������м�

                # ���������͹��˼�
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("fea_encoder"):
                        new_key = k[len("fea_encoder."):]  # �Ƴ� "fea_encoder." ǰ׺
                        new_state_dict[new_key] = v

                # ���ص�ģ����
                msg = model.load_state_dict(new_state_dict, strict=False)
                print(f"=> loaded pre-trained model '{args.pretrained}'")
            else:
                print(f"=> no checkpoint found at '{args.pretrained}'")

        # ������ʧ�������Ż���
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = torch.optim.SGD(list(model.parameters()) + list(classifier.parameters()), args.learning_rate,
                                    weight_decay=args.weight_decay)
        cudnn.benchmark = True

        # ������ǰ�۵����ݼ�����
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
        # ѵ����ǰ��
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch, args)
            
            # ѵ��һ��epoch
            _, _, acc1 = train(fold_train_loader, model, classifier, criterion, optimizer, epoch, args)
            
            # ����֤��������
            acc, probs, preds, current_metrics = validate(fold_val_loader, model, classifier, criterion, args)
        
        # ���浱ǰ�۵���ѽ��
        print(f"\nFold {fold + 1} Results:")
        for metric, value in current_metrics.items():
            metrics_results[metric].append(value)
            print(f"{metric}: {value:.4f}")

        #�����ڴ�
        del model, classifier, optimizer
        torch.cuda.empty_cache()
        
    # ���㲢��ӡ���յ�ƽ�����
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

    # ��ʼ��������������
    all_targets = []
    all_preds = []
    all_probs = []

    # �л�������ģʽ
    model.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for i, ((gpn, hyena), target) in enumerate(val_loader):
            if args.gpu is not None:
                gpn = gpn.cuda(args.gpu, non_blocking=True)
                hyena = hyena.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # �������
            fea = model(gpn, hyena)
            output = classifier(fea)

            # ��ȡԤ����ʺͱ�ǩ
            probs = torch.softmax(output, dim=1)[:, 1]
            preds = (probs > 0.5).long()

            # ��� NaN ֵ������
            if torch.isnan(probs).any() or torch.isnan(preds).any():
                print(f"Warning: NaN values found in batch {i}")
                continue

            # �ռ�����Ԥ���Ŀ��
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # ����ʱ��
            batch_time.update(time.time() - end)
            end = time.time()

    # ת��Ϊnumpy����
    all_targets = np.array(all_targets)  # ��ʵ��ǩ
    all_preds = np.array(all_preds)  # Ԥ���ǩ
    all_probs = np.array(all_probs)  # Ԥ�����

    # ��Ӽ����ȷ������Ч��Ԥ���Ŀ��
    if len(all_targets) == 0 or len(all_preds) == 0:
        print("Error: No valid predictions")
        return 0, [], [], {}

    # �������ָ��
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

    # ��ӡ��ǰ����ָ��
    for metric, value in metrics.items():
        print(f' * {metric}: {value:.4f}')

    return metrics['ACC'], all_probs, all_preds, metrics


def predict(val_loader, model, classifier, criterion, args):
    # �л�������ģʽ
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

            # �������
            fea = model(gpn, hyena)
            output = classifier(fea)

            # ��ȡԤ����ʺͱ�ǩ
            probs = torch.softmax(output, dim=1)[:, 1]
            preds = (probs > 0.5).long()

            # ��� NaN ֵ������
            if torch.isnan(probs).any() or torch.isnan(preds).any():
                print(f"Warning: NaN values found in probabilities or predictions at batch {i}.")
                continue  # ������ǰ����

            # �ռ�����Ԥ���Ŀ��
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # ת��Ϊnumpy����
    all_preds = np.array(all_preds)  # Ԥ���ǩ
    all_probs = np.array(all_probs)  # Ԥ�����

    return all_probs, all_preds  # ��Ȼ����׼ȷ����Ϊ��Ҫָ��


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

    # ��������
    try:
        # ����ѵ����
        if args.high_confi:
            print("\n=> Starting training with 990 samples")
            train_data = pd.read_csv(os.path.join(args.data, "Training_990pos_990neg_start-lost_hg38_1001bp_seq.txt"), sep="\t", header=0)
            print("ѵ����ǩ���سɹ���������״:", train_data.shape)
            train_gpn_msa = torch.load(os.path.join(args.data, "Training_990pos_990neg_start-lost_hg38_GPN-MSA_alt_feature.pth"))
            print("ѵ��GPN-MSA�������سɹ�,������״:", train_gpn_msa.shape)
            train_hyena_dna = torch.load(os.path.join(args.data, "Training_990pos_990neg_start-lost_hg38_1001bp_HyenaDNA_feature.pth"))
            print("ѵ��HyenaDNA�������سɹ�,������״:", train_hyena_dna.shape)
        
        else:
            print("\n=> Starting training with 1264 samples")
            train_data = pd.read_csv(os.path.join(args.data, "1-Training_Label_1264pos_1264neg_start-lost_hg38_1001bp_kmer_random.csv"), sep=",", header=0)
            print("ѵ����ǩ���سɹ���������״:", train_data.shape)
            train_gpn_msa = torch.load(os.path.join(args.data, "1-Training_Label_1264pos_1264neg_start-lost_hg38_GPN-MSA_alt_feature.pth"))
            print("ѵ��GPN-MSA�������سɹ�,������״:", train_gpn_msa.shape)
            train_hyena_dna = torch.load(os.path.join(args.data, "1-Training_Label_1264pos_1264neg_start-lost_hg38_1001bp_HyenaDNA_feature.pth"))
            print("ѵ��HyenaDNA�������سɹ�,������״:", train_hyena_dna.shape)

        # ���ز��Լ�һ
        val_data = pd.read_csv(os.path.join(args.data, "Testing_Label_700pos_486neg_start-lost_hg38_1001bp_kmer_random.txt"), sep="\t", header=0)
        print("���Լ�һ��ǩ���سɹ���������״:", val_data.shape)
        val_gpn_msa = torch.load(os.path.join(args.data, "Testing_Label_700pos_486neg_start-lost_hg38_GPN-MSA_alt_feature.pth"))
        print("���Լ�һGPN-MSA�������سɹ�,������״:", val_gpn_msa.shape)
        val_hyena_dna = torch.load(os.path.join(args.data, "Testing_Label_700pos_486neg_start-lost_hg38_1001bp_random_HyenaDNA_feature.pth"))
        print("���Լ�һHyenaDNA�������سɹ�,������״:", val_hyena_dna.shape)

        # ���ز��Լ���
        test_data = pd.read_csv(os.path.join(args.data, "HGMD-ClinVar_new_startloss_1001bp_seq.txt"), sep="\t", header=0)
        print("���Լ�����ǩ���سɹ���������״:", test_data.shape)
        test_gpn_msa = torch.load(os.path.join(args.data, "HGMD-ClinVar_new_startloss_GPN-MSA_alt_feature.pth"))
        print("���Լ���GPN-MSA�������ݼ��سɹ�,������״:", test_gpn_msa.shape)
        test_hyena_dna = torch.load(os.path.join(args.data, "HGMD-ClinVar_new_startloss_1001bp_HyenaDNA_feature.pth"))
        print("���Լ���HyenaDNA�������ݼ��سɹ�,������״:", test_hyena_dna.shape)

        # ���ز����Ӽ�
        notrain_data = pd.read_csv(os.path.join(args.data, "Testing1_subset_NoTrain_830data_1001bp_seq.txt"), sep="\t", header=0)
        print("δѵ���Ӽ���ǩ���سɹ���������״:", notrain_data.shape)
        notrain_gpn_msa = torch.load(os.path.join(args.data, "Testing1_subset_NoTrain_830data_GPN-MSA_alt_feature.pth"))
        print("δѵ���Ӽ�GPN-MSA�������ݼ��سɹ�,������״:", notrain_gpn_msa.shape)
        notrain_hyena_dna = torch.load(os.path.join(args.data, "Testing1_subset_NoTrain_830data_1001bp_HyenaDNA_feature.pth"))
        print("δѵ���Ӽ�HyenaDNA�������ݼ��سɹ�,������״:", notrain_hyena_dna.shape)

#        # ���ѡ�񲿷��������͸���������ģ��΢��
#        train_data, train_gpn_msa, train_hyena_dna = select_random_samples(args, train_data, train_gpn_msa, train_hyena_dna, sample_ratio=0.1)
#        print("���ѡ���ѵ��������״:", train_data.shape)

#        # �������ݼ�
#        pre_data = pd.read_csv(os.path.join(args.data, "Human_start-lost_list_180350_hg38_1001bp_seq.txt"), sep="\t", header=0)
#        if 'Label' not in pre_data.columns:
#            pre_data['Label'] = -1  # ��� 'Label' �в�����ֵΪ -1
#        print("���ݼ���ǩ���سɹ�,������״:", pre_data.shape)
#        pre_gpn_msa = torch.load(os.path.join(args.data, "Human_start-lost_180350data_GPN-MSA_alt_feature.pth"))
#        print("���ݼ�GPN-MSA�������ݼ��سɹ�, ������״:", pre_gpn_msa.shape)
#        pre_hyena_dna = torch.load(os.path.join(args.data, "Human_start-lost_180350data_1001bp_HyenaDNA_feature.pth"))
#        print("���ݼ�HyenaDNA�������ݼ��سɹ�,������״:", pre_hyena_dna.shape)

        # �������ݼ�
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

        notrain_dataset = LinearClassifierDataset(
            notrain_data,
            notrain_gpn_msa,
            notrain_hyena_dna,
            train=False
        )

#        pre_dataset = LinearClassifierDataset(
#            pre_data,
#            pre_gpn_msa,
#            pre_hyena_dna,
#            train=False
#        )

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # �������ݼ�����
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

    notrain_loader = DataLoader(
        notrain_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

#    pre_loader = DataLoader(
#        pre_dataset,
#        batch_size=args.batch_size,
#        shuffle=False,
#        num_workers=args.workers,
#        pin_memory=True
#    )

    # ���н�����֤
    if args.cross_validation:
        print("\n=> Starting five-fold cross validation")
        # ����5�۽�����֤
        metrics_results = cross_validate(train_dataset, args)

    else:
        print("\n=> Starting training with whole training set")
        # ����ģ��
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

        # ����Ԥѵ��ģ��
        if args.pretrained:
            if os.path.isfile(args.pretrained):
                print(f"=> loading checkpoint '{args.pretrained}'")
                checkpoint = torch.load(args.pretrained, map_location="cpu")

                # ��ȡԤѵ���� state_dict
                state_dict = checkpoint['state_dict']
                print("State dict keys:", state_dict.keys())  # ������м�

                # ���������͹��˼�
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("fea_encoder"):
                        new_key = k[len("fea_encoder."):]  # �Ƴ� "fea_encoder." ǰ׺
                        new_state_dict[new_key] = v

                # ���ص�ģ����
                msg = model.load_state_dict(new_state_dict, strict=False)
                print(f"=> loaded pre-trained model '{args.pretrained}'")
            else:
                print(f"=> no checkpoint found at '{args.pretrained}'")

        # ������ʧ�������Ż���
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = torch.optim.SGD(list(model.parameters()) + list(classifier.parameters()), args.learning_rate,
                                    weight_decay=args.weight_decay)
        cudnn.benchmark = True

        # ��ѡ���Ӽ���ָ�
        best_acc1 = 0
        to_restore = {"epoch": 0}
        start_epoch = to_restore["epoch"]

        # ѵ��ѭ��
        for epoch in range(start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args)

            # ѵ��һ��epoch
            _, _, acc1 = train(train_loader, model, classifier, criterion, optimizer, epoch, args)

            # �ڲ��Լ�һ������
            print("\n�ڲ��Լ�һ������:")
            validate(val_loader, model, classifier, criterion, args)

            # ��¼���acc���������
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
                filename=f"simCLR_finetune_{epoch:04d}.pth.tar"
            )


        # �ڲ��Լ�һ�Ͻ�����������
        # print("\n�ڲ��Լ�һ������:")
        # validate(val_loader, model, classifier, criterion, args)

        # �ڲ��Լ����Ͻ�����������
        print("\n�ڲ��Լ���������:")
        validate(test_loader, model, classifier, criterion, args)

        # ��δѵ���Ӽ��Ͻ�����������
        print("\n��δѵ���Ӽ�������:")
        validate(notrain_loader, model, classifier, criterion, args)


#        # �����ݼ��Ͻ���Ԥ�Ⲣ���
#        print("\n���Ԥ����:")
#        pred_score, pred_label = predict(pre_loader, model, classifier, criterion, args)
#        pred_score_df = pd.DataFrame(pred_score, columns=['pred_score'])  # ��Ԥ�����ת��Ϊ DataFrame
#        pred_label_df = pd.DataFrame(pred_label, columns=['pred_label'])  # ��Ԥ���ǩת��Ϊ DataFrame
#        merged_data = pd.concat([pre_data.iloc[:, :8], pred_score_df, pred_label_df], axis=1)
#        output_path = "./Human_start-lost_180350data_startCLR_pred_result.txt"
#        merged_data.to_csv(output_path, sep="\t", index=False)
#        print(f"���ݼ�Ԥ�����ѱ�����: {output_path}")

if __name__ == '__main__':
    main()