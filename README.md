# PRO-LDM

---

![image](https://github.com/AzusaXuan/PRO-LDM/blob/main/proldm_1.png)

## Introduction

Here we present PRO-LDM: a modular multi-tasking framework combining design fidelity and computational efficiency, by integrating the diffusion model in latent space to perform protein design. 

PRO-LDM 

+ learns biological representations from natural proteins from both amino acids and whole sequence at local and global levels

+ designs natural-like new sequences with enhanced diversity

+ conditionally designs new proteins with tailored properties or functions

Out-of-distribution (OOD) design can be implemented by adjusting the classifier-free guidance that enables PRO-LDM to sample notably different regions in the latent space for functional optimization of natural proteins.



---



## Environment Setup

~~~python
# create new environment
conda create -n proldm_env python=3.8
conda activate proldm_env
# prepare for new environment by self is recommmended, or you can use the following command
pip3 install -r requirements.txt
~~~



---



## How to Run

### Pre-train Models

set 

~~~python
args.mode=train 
~~~

to train models,

you can set 

~~~
args.multi_gpu=True 
~~~

to use DataParallel Training, 

use 

~~~
args.device_id 
~~~

to set GPU number.

~~~bash
# 4-GPU training
python main.py --mode train --dataset <dataset_name> --multi_gpu True --device_id [0, 1, 2, 3]
# 1-GPU training
python main.py --mode train --dataset <dataset_name> --multi_gpu False --device_id [0]
~~~

~~~bash
# available args.dataset and corresponding args.num_labels
# conditional datasets
# n-label means you can choose dif_sample_label=0 for unconditional sample, and dif_sample_label=1~n for conditional sample
5-label: NESP, ube4b
8-label: gifford, GFP, TAPE, pab1, bgl3, HIS7, CAPSD, B1LPA6
# unconditional datasets
0-label: MSA, MSA_RAW, MDH
~~~



#### Unconditional Training

~~~bash
# for example: train unconditional dataset MDH
python main.py --mode train --dataset MDH -n_epochs 1000
# ckpt is saved in path: PROLDM_OUTLIER/train_logs/MDH
~~~



#### Conditional Training

~~~bash
# train conditional dataset TAPE
python main.py --mode train --dataset TAPE --n_epochs 1000
# ckpt is saved in path: PROLDM_OUTLIER/train_logs/TAPE
# expected running time: ~5 hours on 4*V100
~~~



### Evaluate Models

we use dataset TAPE as an example, and the result is generated in path: **PROLDM_OUTLIER/test_output/TAPE**

~~~bash
python main.py --mode eval --dataset <dataset_name> --eval_load_epoch <eval_epoch>
# for example
python main.py --mode eval --dataset TAPE --eval_load_epoch 1000
~~~

### Seq Generation

we use dataset TAPE as an example, and the result is generated in path: **PROLDM_OUTLIER/generated_seq/TAPE**

Label 0 represents unconditional generation, and label 1-8 represent conditional generation.

~~~sh
python main.py --mode sample --dataset <dataset_name> --dif_sample_label <label> --dif_sample_epoch <epoch>
# for example
python main.py --mode sample --dataset TAPE --dif_sample_label 0 --dif_sample_epoch 1000
~~~

### Others

you can also use .sh file to run the code

~~~sh
#/bin/sh
#BSUB -m node01-10
#BSUB -J PROLDM
#BSUB -n 8
#BSUB -W 7200
#BSUB -gpu "num=8"
#BSUB -o out.txt
#BSUB -e err.txt

module load anaconda3
module load cuda/11.6
source activate proldm_env

python main.py --mode train --dataset TAPE
~~~



---

### You can access all raw code, checkpoints and data via the following command in terminal.

~~~
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1tX9PSrywPhW62HlExn3lWj2IGEsSzKFv
~~~

## Model Checkpoint

we save pre-trained model weights in the folder **./train_logs**

---



## Training Datasets

All training datasets are preserved in the folder **./data** and most of them are divided into two csv files: **<datset_name>-train.csv** and **<dataset_name>-test.csv**.



---



