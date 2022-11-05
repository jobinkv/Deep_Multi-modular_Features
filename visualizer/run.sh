#!/bin/bash
#SBATCH -A jobinkv
#SBATCH -c 4
#       #SBATCH --reservation non-deadline-queue
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --profile=<all|none|[energy[,|task[,|filesystem[,|network]]]]>


echo "Running on: $SLURM_NODELIST"
mkdir -p /ssd_scratch/cvit/jobinkv
mkdir -p /ssd_scratch/cvit/jobinkv/pyTorchPreTrainedModels
cd /ssd_scratch/cvit/jobinkv
# geting the image
rsync -avz jobinkv@10.2.16.142:/mnt/1/scriptIdentificationDataset/* /ssd_scratch/cvit/jobinkv
rsync -avz jobinkv@10.2.16.142:/mnt/4/ijdarTrainedModels/epoch_20_loss_1.14508_testAcc_0.97219_ged_resnext101script_script_K_20_G_64_C_256.pth /ssd_scratch/cvit/pyTorchPreTrainedModels/resnext101script.pth

cd /home/jobinkv/ijdarWork19/classiication/code/visualizer/
 
python main.py --trainedModel /ssd_scratch/cvit/pyTorchPreTrainedModels/resnext101script.pth --dataset script -k 20 -g 64 -c 256 

