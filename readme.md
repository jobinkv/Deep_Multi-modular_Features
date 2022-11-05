# Document Image Analysis using Deep Multi-modular Features
## Official Project Webpage
A deep network architecture that independently
learns texture patterns, discriminative patches, and shapes to solve
various document image analysis tasks. [PDF](https://link.springer.com/article/10.1007/s42979-022-01414-4)

<p align="center">
<img src="imgs/propoesArch.jpg" />
<em>
<p>
Presents a block diagram of the proposed approach. The input image passes through the layers of convolutional filters of an CNN architecture to extract convolutional features. From the convolutional feature, the model extracts three different modalities of features: an encoding feature, a global feature, and a discriminative feature.
</p></em>

This repository provides the official PyTorch implementation of the Journal:
> **Document Image Analysis using Deep Multi-modular Features** <br>
> Jobin K.V., Ajoy Mondal, and C. V. Jawahar<br>
> In SNCS 2022<br>
>[PDF](https://cvit.iiit.ac.in/images//JournalPublications/2022/Multi_modular.pdf)

> **Abstract:** *
Texture or repeating patterns, discriminative patches, and shapes are the salient features for various document image analysis problems. This article proposes a deep network architecture that independently learns texture patterns, discriminative patches, and shapes to solve various document image analysis tasks. The considered tasks are document image classification, genre identification from book covers, scientific document figure classification, and script identification. The presented network learns global, texture, and discriminative features and combines them judicially based on the nature of the problems to be solved. We compare the performance of the proposed approach with state-of-the-art techniques on multiple publicly available datasets such as Book-Cover, RVL-CDIP, CVSI and DocFigure. Experiments show that our approach outperforms genre and document figure classifications more than state-of-the-art and obtains comparable results on document image and script classification tasks.
*<br>

## Pytorch Implementation
### Installation
```
conda create --name dmmf python=3.8
conda activate dmmf
conda install -y pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
```

install pytorch encoding from [here](https://hangzhang.org/PyTorch-Encoding/notes/compile.html)
<br>
To check the installation of pytorch encoding
<br>
run in python console
```
import encoding
```

Install following packages.
```
conda install scipy==1.4.1
conda install tqdm==4.46.0
conda install scikit-image==0.16.2
pip install tensorboardX==2.0
pip install thop

git clone https://github.com/jobinkv/Deep_Multi-modular_Features.git
cd Deep_Multi-modular_Features
```



=====END=========
```



module load u18/cudnn/8.3.3-cuda-10.2 u18/cuda/10.2
conda activate slide
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install git+https://github.com/openai/CLIP.git
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html


```
