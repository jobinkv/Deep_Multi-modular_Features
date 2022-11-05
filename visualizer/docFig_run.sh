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
cd /ssd_scratch/cvit/jobinkv/docfig

# geting the image
if [ ! -f "/ssd_scratch/cvit/jobinkv/docfig/images/2008_04587413-Figure3-1subFig-1.png" ]
then
echo "Files not found, copying please waite!"
rsync -avz jobinkv@10.2.16.142:/mnt/3/subFigDataset/grec_19.zip /ssd_scratch/cvit/jobinkv/docfig
rsync -avz jobinkv@10.2.16.142:/mnt/4/datas/Thesis/dataset/grec_19_dataset/annotation /ssd_scratch/cvit/jobinkv/docfig/
unzip grec_19.zip
else
        echo "All the file are avilable!"
fi

rsync -avz jobinkv@10.2.16.142:/mnt/4/ijdarTrainedModels/epoch_15_loss_1.27678_testAcc_0.95909_ged_resnext101docSeg_docSeg_K_20_G_64_C_256.pth /ssd_scratch/cvit/pyTorchPreTrainedModels/resnext101docSeg.pth

cd /home/jobinkv/ijdarWork19/classiication/code/visualizer/
 
python main.py --trainedModel /ssd_scratch/cvit/pyTorchPreTrainedModels/resnext101docSeg.pth \
  --dataset docSeg -k 20 -g 64 -c 256 --selectedFile listdocfig.txt 

