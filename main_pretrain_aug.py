#!/usr/bin/env python
# coding=gbk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
import time
import math
import shutil
import warnings
from tqdm import tqdm
import copy
from models.StartUp import StartUp
import argparse


# �Զ������ݼ��ࣨ���䣩
class ContrastiveDataset(Dataset):
    def __init__(self, data, gpn_msa, hyena_dna, labels=None):
        # ȷ��data����ȷ����ֵ����
        self.data = data
        self.gpn_msa = gpn_msa
        self.hyena_dna = hyena_dna
        # ���û���ṩlabels������Ĭ������Ϊһ��������data��ͬ���б��б�����ȫΪ-1
        self.labels = labels if labels is not None else [-1] * len(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            # ��ȡ��������������
            data = self.data[idx]
            gpn_msa = self.gpn_msa[idx]
            hyena_dna = self.hyena_dna[idx]
            label = self.labels[idx]

            # ��������ת��
            if isinstance(data, np.ndarray):
                # ȷ�������Ǹ�����
                if np.issubdtype(data.dtype, np.number):
                    data = data.astype(np.float32)
                else:
                    # ������ַ������飬������ȡ��ֵ
                    try:
                        data = np.array([float(x) for x in data if isinstance(x, (int, float))])
                    except ValueError:
                        raise ValueError(f"Invalid data type for data: {data}")
                data = torch.from_numpy(data)
            else:
                # ����ǵ���ֵ��ת��Ϊ������
                try:
                    data = float(data)
                except ValueError:
                    raise ValueError(f"Could not convert data to float: {data}")
                data = torch.tensor([data], dtype=torch.float32)

            # ת��GPN��Hyena����Ϊtensor
            gpn_msa = torch.tensor(gpn_msa, dtype=torch.float32)
            hyena_dna = torch.tensor(hyena_dna, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.long)

            return data, gpn_msa, hyena_dna, label

        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            print(f"Data type: {type(self.data[idx])}")
            print(f"Data value: {self.data[idx]}")  # ������Ϣ
            raise


class TokenCutoff:
    def __init__(self, cutoff_rate=0.15, mode="random", seed=None):
        self.cutoff_rate = cutoff_rate
        self.mode = mode
        self.seed = seed  # ���Դ�������

        # ������ӱ����ã������������ʼ��
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    def apply(self, embeddings, attention_scores=None):

        batch_size, seq_len, embedding_dim = embeddings.shape
        num_tokens_to_cutoff = int(seq_len * self.cutoff_rate)

        cutoff_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=embeddings.device)

        if self.mode == "random":
            for i in range(batch_size):
                # ��������ʹ�� np.random.choice �����ѡ�� tokens
                indices = np.random.choice(seq_len, num_tokens_to_cutoff, replace=False)
                cutoff_mask[i, indices] = False

        embeddings[~cutoff_mask] = 0
        return embeddings


class FeatureCutoff:
    def __init__(self, cutoff_rate=0.15, mode="random", seed=None):
        self.cutoff_rate = cutoff_rate
        self.mode = mode
        self.seed = seed  # ���Դ�������

        # ������ӱ����ã������������ʼ��
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    def apply(self, embeddings, attention_scores=None):

        batch_size, seq_len, embedding_dim = embeddings.shape
        num_features_to_cutoff = int(embedding_dim * self.cutoff_rate)

        cutoff_mask = torch.ones(batch_size, seq_len, embedding_dim, dtype=torch.bool, device=embeddings.device)

        if self.mode == "random":
            # ���ѡ����Ҫ�ڱε�����
            for i in range(batch_size):
                # ��ÿ��������������ڱβ���
                indices = torch.randint(0, embedding_dim, (num_features_to_cutoff,), device=embeddings.device)
                cutoff_mask[i, :, indices] = False

        embeddings[~cutoff_mask] = 0
        return embeddings


class RandomMask:
    def __init__(self, dropout_ratio=0.5, mode="random", seed=None):
        self.dropout_ratio = dropout_ratio
        self.mode = mode
        self.seed = seed  # ���Դ�������

        # ������ӱ����ã������������ʼ��
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    def apply(self, embeddings, attention_scores=None):

        batch_size, seq_len, embedding_dim = embeddings.shape
        total_features = seq_len * embedding_dim  # ����ÿ����������������
        num_features_to_cutoff = int(total_features * self.dropout_ratio)

        cutoff_mask = torch.ones_like(embeddings, dtype=torch.bool, device=embeddings.device)

        if self.mode == "random":
            # ���ѡ����Ҫ�ڱε�����
            for i in range(batch_size):
                # Ϊÿ�������������λ��
                flat_indices = torch.randperm(total_features, device=embeddings.device)[:num_features_to_cutoff]
                # ��һά����ת��Ϊ��ά����
                row_indices = flat_indices // embedding_dim
                col_indices = flat_indices % embedding_dim
                # �ڶ�Ӧλ������mask
                cutoff_mask[i, row_indices, col_indices] = False

        embeddings[~cutoff_mask] = 0
        return embeddings


class proj_head(nn.Module):
    def __init__(self, input_length):
        super(proj_head, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_length, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 64),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return logits, labels, loss / (2 * self.batch_size)


# SimCLRģ����
class SimCLR(nn.Module):
    def __init__(self, device, base_encoder, args, temperature=0.07):
        super(SimCLR, self).__init__()

        self.temperature = temperature

        self.fea_encoder = base_encoder(gpn_embedding_dim=args.gpn_dim, hyena_embedding_dim=args.hyena_dim,
                                        n_filters=args.n_filters, filter_sizes=args.filter_sizes, output_dim=2,
                                        dropout=args.dropout)  # ������
        self.proj_head = proj_head(args.fea_dim)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
        self.contrastive_loss = NTXentLoss(device, args.batch_size, args.temperature, True)

    def forward(self, gpn_q, gpn_k, hyena_q, hyena_k, device):
        # ȷ��������������ȷ���豸��
        gpn_q = gpn_q.to(device)
        gpn_k = gpn_k.to(device)
        hyena_q = hyena_q.to(device)
        hyena_k = hyena_k.to(device)
        
        feature1 = self.fea_encoder(gpn_q, hyena_q)
        z1 = self.proj_head(feature1)

        feature2 = self.fea_encoder(gpn_k, hyena_k)
        z2 = self.proj_head(feature2)

        # normalize projection feature vectors
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Cross-Entropy loss
        logits, labels, loss = self.contrastive_loss(z1, z2)

        return logits, labels, loss


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


def accuracy(output, target, topk=(1,)):
    """����topk׼ȷ��"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))  # �޸�Ϊ reshape

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # �޸�Ϊ reshape
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, optimizer, epoch, args, device):
    """ѵ��һ��epoch"""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    loss_meter = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # ��ӽ�������ʾ
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_meter, top1, top5],
        prefix=f"Epoch: [{epoch}/{args.epochs}]"  # �����������ʾ
    )

    model.train()
    total_loss = 0
    total_top1 = 0
    total_top5 = 0
    end = time.time()

    for i, (_, gpn_batch, hyena_batch, _) in enumerate(train_loader):
        try:
            data_time.update(time.time() - end)
            batch_size = gpn_batch.size(0)
            
            # �������ƶ���GPU
            gpn_batch = gpn_batch.to(device)
            hyena_batch = hyena_batch.to(device)

            # �洢��ǿ�������
            gpn_q_list = []
            gpn_k_list = []
            hyena_q_list = []
            hyena_k_list = []

            # ��ÿ����������������ǿ
            for j in range(batch_size):
                # ��ȡ��������
                gpn_sample = gpn_batch[j].unsqueeze(0)  # ���ά�ȣ���״: [1, seq_len, dim]
                hyena_sample = hyena_batch[j].unsqueeze(0)

                # ��ͬһ�������������β�ͬ���������
                #cutoff1 = TokenCutoff(cutoff_rate=0.25, mode="random", seed=args.seed + epoch * batch_size + j)
                #cutoff2 = TokenCutoff(cutoff_rate=0.25, mode="random", seed=args.seed + epoch * batch_size + j + 222)

                #cutoff1 = FeatureCutoff(cutoff_rate=0.25, mode="random", seed=args.seed + epoch * batch_size + j)
                #cutoff2 = FeatureCutoff(cutoff_rate=0.25, mode="random", seed=args.seed + epoch * batch_size + j + 222)
                
                cutoff1 = RandomMask(dropout_ratio=0.25, mode="random", seed=args.seed + epoch * batch_size + j)
                cutoff2 = RandomMask(dropout_ratio=0.25, mode="random", seed=args.seed + epoch * batch_size + j + 222)

                # �Ե�����������������ǿ��������������
                gpn_q = cutoff1.apply(gpn_sample)
                gpn_k = cutoff2.apply(gpn_sample)
                hyena_q = cutoff1.apply(hyena_sample)
                hyena_k = cutoff2.apply(hyena_sample)

                # ����ǿ���������ӵ��б���
                gpn_q_list.append(gpn_q)
                gpn_k_list.append(gpn_k)
                hyena_q_list.append(hyena_q)
                hyena_k_list.append(hyena_k)

            # ���б�ת��Ϊ��������
            gpn_q_batch = torch.cat(gpn_q_list, dim=0)
            gpn_k_batch = torch.cat(gpn_k_list, dim=0)
            hyena_q_batch = torch.cat(hyena_q_list, dim=0)
            hyena_k_batch = torch.cat(hyena_k_list, dim=0)                

            #����ݶ�
            optimizer.zero_grad()
            
            # �������
            output, target, loss = model(gpn_q_batch, gpn_k_batch, hyena_q_batch, hyena_k_batch, device)

            # ����׼ȷ��
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            loss_meter.update(loss.item(), gpn_q.size(0))
            top1.update(acc1[0], gpn_q.size(0))
            top5.update(acc5[0], gpn_q.size(0))

            # ���򴫲����Ż�
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_top1 += acc1[0]
            total_top5 += acc5[0]

            # ��������ʱ��
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        except Exception as e:
            print(f"Error in training step: {e}")
            continue

    avg_loss = total_loss / len(train_loader)
    avg_acc1 = total_top1 / len(train_loader)

    return avg_loss, avg_acc1


# ѧϰ�ʵ�������
def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:
        # ʹ�������˻����ѧϰ��
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:
        # ʹ��Ԥ�������̱�����ѧϰ��
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class ProgressMeter:
    """��ʾѵ������"""

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


def save_checkpoint(state, is_best, save_dir="./pretrain", filename="checkpoint.pth.tar"):
    # ��Ӱ汾���ƺͶ�������
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, filename)

    # ��Ӵ�����
    try:
        torch.save(state, filepath)
        if is_best:
            best_filepath = os.path.join(save_dir, 'model_best.pth.tar')
            shutil.copyfile(filepath, best_filepath)
            print(f"Saved best model to {best_filepath}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def main():
    parser = argparse.ArgumentParser(description="MoCo Pretraining")
    # �����в���
    parser.add_argument("--data", metavar="DIR", default="./dataset", help="path to dataset")
    parser.add_argument("--epochs", default=20, type=int, help="number of total epochs to run")
    parser.add_argument("--batch_size", default=256, type=int, help="mini-batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="initial learning rate")
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument("--seed", default=666, type=int, help="seed for initializing training")
    parser.add_argument("--gpu", default=1, type=int, help="GPU id to use")
    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")
    parser.add_argument("--gpn_dim", default=768, type=int, help="GPN embedding dimension")
    parser.add_argument("--hyena_dim", default=128, type=int, help="HyenaDNA embedding dimension")
    parser.add_argument("--num_id_pos", default=6414, type=int, help="number of ID positions")
    parser.add_argument("--n_filters", default=64, type=int, help="number of filters")
    parser.add_argument("--filter_sizes", default=[3, 4, 5], type=int, nargs="+", help="filter sizes")
    parser.add_argument("--fea_dim", default=1024, type=int, help="output dimension")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout rate")
    parser.add_argument("--temperature", default=0.07, type=float, help="softmax temperature")
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    parser.add_argument("--model_save_dir", default="./pretrain", type=str, help="path to save checkpoints")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta1")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta2")
    args = parser.parse_args()

    # �Ż�GPU����
    if torch.cuda.is_available():
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            device = torch.device(f'cuda:{args.gpu}')
            torch.backends.cudnn.benchmark = True  # ����������������
        else:
            device = torch.device('cuda:0')
            warnings.warn('No GPU specified, using default GPU 0')
    else:
        device = torch.device('cpu')
        warnings.warn('No GPU available, using CPU for training!')

    args.device = device

    # ������������Ա�֤���ظ���
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    # ����GPU�豸
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ��������
    try:
        data = pd.read_csv(os.path.join(args.data, "gnomADv3_22981data_unlabeled_1001bp_seq.txt"), sep="\t", header=0)
        print("���ݼ��سɹ���������״:", data.shape)  # ��ӵ�����Ϣ
        # ��������ת��
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].astype(np.float32)
        # ���������ļ�
        gpn_msa = torch.load(os.path.join(args.data, "gnomADv3_22981data_unlabeled_GPN-MSA_alt_feature.pth"))
        hyena_dna = torch.load(os.path.join(args.data, "gnomADv3_22981data_unlabeled_1001bp_HyenaDNA_feature.pth"))
        print("�����ļ����سɹ�")  # ��ӵ�����Ϣ

        # ȷ������������ȷ
        if not isinstance(gpn_msa, torch.Tensor):
            gpn_msa = torch.tensor(gpn_msa, dtype=torch.float32)
        if not isinstance(hyena_dna, torch.Tensor):
            hyena_dna = torch.tensor(hyena_dna, dtype=torch.float32)

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    train_dataset = ContrastiveDataset(
        data=data.values,  # ת��Ϊnumpy����
        gpn_msa=gpn_msa,
        hyena_dna=hyena_dna
    )
    # ���ݼ���������
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count()),  # ����CPU��������̬����
        pin_memory=True if torch.cuda.is_available() else False,  # �����Ƿ���GPU����
        drop_last=True,
        persistent_workers=True if os.cpu_count() > 1 else False,  # ����CPU��������
        prefetch_factor=2  # ���Ԥȡ����
    )

    # ģ�ͳ�ʼ��
    model = SimCLR(device=device, base_encoder=StartUp, args=args, temperature=args.temperature).to(args.device)
    # ʹ�� NTXentLoss ��Ϊ��ʧ����
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # �Ӽ���ָ���������ڣ�
    best_loss = float('inf')
    best_acc = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=args.device)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    print("=> starting training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # ѵ��һ��epoch
        avg_loss, avg_acc = train(train_loader, model, optimizer, epoch, args, args.device)

        # ͬʱ������ʧ��׼ȷ��
        is_best = avg_loss < best_loss or (avg_loss == best_loss and avg_acc > best_acc)
        best_loss = min(avg_loss, best_loss)
        best_acc = max(avg_acc, best_acc) if avg_loss == best_loss else best_acc

        # �������
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            is_best=False,
            save_dir=args.model_save_dir,
            filename="checkpoint_{:04d}.pth.tar".format(epoch)
        )

        # ��ӡ��ǰѵ��״̬
        print(f'Epoch: [{epoch + 1}/{args.epochs}]\t'
              f'Loss: {avg_loss:.4f}\t'
              f'Acc@1: {avg_acc:.2f}%\t'
              f'Best Loss: {best_loss:.4f}\t'
              f'Best Acc: {best_acc:.2f}%')

    elapsed_time = time.time() - start_time
    print(f"Training complete. Total time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()