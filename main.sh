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
source activate relso_repo_env


python main.py --mode train --dataset TAPE