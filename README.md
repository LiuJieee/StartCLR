# Prediction of human pathogenic start loss variants based on self-supervised contrastive learning

StartCLR is a novel prediction method that combines self-supervised contrastive learning pre-training with supervised fine-tuning specifically for identifying pathogenic start loss variants. It effectively utilizes information from a large amount of unlabeled data and a small amount of labeled data, integrates embedding features from different DNA language models to comprehensively characterize the variant context sequence, and thus achieves accurate prediction of pathogenic variants.
![Main Figure](./figs/StartCLR_flowchart.png)

## Basic requirements
To install dependencies, create a new conda environment:
```bash
conda env create -f StartCLR.yml
```
We run the program on the Ubuntu 22.04.4 LTS system.

## Additional requirements for model construction

### GPN-MSA
This module is used to quantify the embedding features of mutated sequence derived from GPN-MSA.

To install dependencies, create a new conda environment:
```bash
pip install git+https://github.com/songlab-cal/gpn.git
conda env create -f GPN-MSA.yml
```
To download the pre-trained model：
```bash
cd GPNMSA
wget https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip
cd model
wget https://huggingface.co/songlab/gpn-msa-sapiens/resolve/main/pytorch_model.bin?download=true
```
For more information about GPN-MSA, see https://doi.org/10.1101/2023.10.10.561776 and https://github.com/songlab-cal/gpn.

### HyenaDNA 
This module is used to quantify the embedding features of mutated sequence derived from HyenaDNA.

To install dependencies, create a new conda environment:
```bash
cd HyenaDNA
conda create -n hyena-dna python=3.8
conda activate hyana-dna
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt

# install Flash Attention
cd HyenaDNA/flash-attention
git clone --recurse-submodules https://github.com/Dao-AILab/flash-attention.git

# loading the pre-trained model 
cd HuggingFace/hyenadna-tiny-1k-seqlen
wget https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen/resolve/main/.gitattributes?download=true
wget https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen/resolve/main/README.md?download=true
wget https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen/resolve/main/config.json?download=true
wget https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen/resolve/main/weights.ckpt?download=true
```
For more information about HyenaDNA, see https://proceedings.neurips.cc/paper_files/paper/2023/hash/86ab6927ee4ae9bde4247793c46797c7-Abstract-Conference.html and https://github.com/HazyResearch/hyena-dna.

## Quick start

### Input format
StartCLR supports variants in TXT format as input. The input file should contain at least 6 columns in the header as follows. [Sample file](./dataset/sample_file.txt)

|  Chr  | Pos |  Ref  |  Alt  |  Label  |  Mutated sequence  |  ...  |
| ----- | --- | ----- | ----- | ------- | ------------------ | ----- |

Please note that the length of the mutated sequences are 1001 base pairs(bp), with the mutation site at the center, and the context sequences on each side are 500 bp. There is no LABEL column in the unlabeled data used for pre-training.

### Input file process
```bash
cd StartCLR
Rscript data_process.R
```
In this section, the example output file titled 'sample_GPNMSA_input.csv' and 'sample_HyenaDNA_input.txt'.

### Quantify the embedding feature based on GPN-MSA
```bash
conda activate GPN-MSA
cd GPNMSA
python GPN-MSA_feature_prepare.py
```
In this section, the example output file titled 'sample_GPN-MSA_feature.pth'.

### Quantify the embedding feature based on HyenaDNA
```bash
conda activate hyena-dna
cd HyenaDNA
python HyenaDNA_feature_prepare.py
```

In this section, the example output file titled 'sample_HyenaDNA_features.pth'.

### Pathogenicity prediction

#### Encoder pre-training
```bash
conda activate StartCLR
cd StartCLR
python main_pretrain_aug.py  --epochs 20  --batch_size 256  --lr 0.0001  --dropout 0.1
```
#### Encoder fine-tuning and classifier training
```bash
python main_finetune_aug.py   --epochs 20   --learning_rate 0.5  --dropout 0.1  --pretrained ./pretrain/checkpoint_0019.pth.tar
```
The training and test datasets of StartCLR are available in the file all_dataset.zip, which can be downloaded from https://zenodo.org/records/13689721. Additionally, the pre-trained encoder and fine-tuned model are also available for download at the same link.
* Due to restrictions associated with the HGMD Professional database, we do not provide the complete information on labeled start loss variants. For details, please refer to the Data Collection and Processing section of the paper.

### Output format
The program directly displays the prediction results upon completion.

It provides a comprehensive overview of StartCLR's predictive performance on the dataset, encompassing various metrics such as recall, specificity (SPE), precision (PRE), F1-score (F1), Matthew's correlation coefficient (MCC), accuracy (ACC), the area under the receiver operating characteristic curve (AUC), and the area under the precision-recall curve (AUPR).
The scoring threshold for StartCLR is established at 0.5, whereby variants scoring below 0.5 are designated as benign and those scoring above 0.5 are identified as pathogenic.

## Cite us
```
@misc{
      title={Prediction of human pathogenic start loss variants based on self-supervised contrastive learning}, 
      author={Jie Liu and Henghui Fan and Na Cheng and Yansen Su and Junfeng Xia},
      year={2025}
}
```
