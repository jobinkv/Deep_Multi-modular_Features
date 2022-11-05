import os
import ipdb
from PIL import Image, ImageOps
import random

# classes
class multiscaleImg:
    def __init__(self,rsize,baseHeights):
        assert sum(baseHeights)<=rsize
        self.rsize = rsize
        self.bheights = baseHeights

    def __call__(self,image):
        new_im = Image.new('RGB', (self.rsize,self.rsize))
        i_cords = 0
        for i in self.bheights :
            scale = (i/float(image.size[1]))
            wsize = int((float(image.size[0])*float(scale)))
            image1 = image.resize((wsize,i), Image.ANTIALIAS)
            for j in range(0, self.rsize, wsize):
                new_im.paste(image1, (j, i_cords))
            i_cords +=i
        return new_im


if __name__ == '__main__':
    # inputs
    data_path = '/ssd_scratch/cvit/jobinkv/cvsi2015/GroundTruth_TestDataset_CVSI2015/Task4'
    imag_list = os.listdir(data_path)
    out_path    = '/ssd_scratch/cvit/jobinkv/scriptPPOut'
    transform = multiscaleImg(384,[100,40,80,160])
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    for item in imag_list:
        if '.jpg' in item:
            image = Image.open(os.path.join(data_path,item)).convert('L').convert('RGB')
            if random.random()>0.5:
                image = ImageOps.invert(image)
            out_img = transform(image)
            out_img.save(os.path.join(out_path,item))
            print (item)

    
