import torch
import sys
from torch.autograd import Variable
import numpy as np
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize
import glob
import os
from pathlib import Path
import cv2

#remove all depth maps which are almost white
sequence_number = '0008'
root = sequence_number + '/dense0/processed/depth/'
image_files = glob.glob(root+'/*.png')
for i, file_path in enumerate(image_files):
    image = cv2.imread(file_path)
    file_name = os.path.split(file_path)[-1]
    mean = image.mean()
    if mean > 230:
        print('deleting ', file_name,image.mean())
        os.remove(file_path)
        image_path = file_path.replace('depth','images').replace('png','jpg')
        os.remove(image_path)

