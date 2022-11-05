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
rsync -avz jobinkv@10.2.16.142:/mnt/1/book-dataset/Task1 /ssd_scratch/cvit/jobinkv/
rsync -avz jobinkv@10.2.16.142:/mnt/4/ijdarTrainedModels/epoch_14_loss_1.67548_testAcc_0.34305_ged_resnext101book_cover_book_cover_K_20_G_64_C_512.pth /ssd_scratch/cvit/jobinkv/pyTorchPreTrainedModels/resnext101book_cover.pth

cd /home/jobinkv/ijdarWork19/classiication/code/visualizer/
 
python main.py --trainedModel /ssd_scratch/cvit/jobinkv/pyTorchPreTrainedModels/resnext101book_cover.pth \
 --dataset book_cover -k 20 -g 64 -c 512

