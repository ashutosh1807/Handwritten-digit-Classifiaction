import torch
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image
import os

dataset_path = './assignment-1-dataset/dataset'
gt_path = './assignment-1-dataset/groundTruthValues.txt'

save_root = './dataset'
src_root = dataset_path


text = open(gt_path)
lines = text.readlines()
index = 0 

prev_img = -1
for _, line in enumerate(lines[0:]):
    img_name = line.split(',')[0]
    if(prev_img != img_name or prev_img == -1):
        prev_img = img_name
        linenumber = 0
    str_gt = line.split(',')[1].split('\n')[0]
    if(str_gt != 'XXXXXXXXXX'):
      img = Image.open(os.path.join(src_root, img_name + '.png'))
      for column_number, gt in enumerate(str_gt):
          if(gt != 'X'):
            bbox = (32*column_number, 32*linenumber, 32*(column_number+1), 32*(linenumber+1))
            img_c = img.crop(bbox) # 32*32 
            print(os.path.join(save_root, img_name + '_'+ str(linenumber) + '_' + str(column_number) + '.png'),gt, sep = ',')
            
            img_c.save(os.path.join(save_root, img_name + '_'+ str(linenumber) + '_' + str(column_number) + '.png'))
    linenumber += 1