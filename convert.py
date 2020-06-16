import torch
import sys
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize
import glob
import os
from pathlib import Path
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = create_model(opt)

input_height = 384
input_width  = 512


def test_simple(model):
    total_loss =0 
    toal_count = 0
    print("============================= TEST ============================")
    model.switch_to_eval()
    root = opt.sequence_number + '/dense0/imgs/'
    Path(opt.sequence_number + "/dense0/processed/depth/").mkdir(parents=True, exist_ok=True)
    Path(opt.sequence_number + "/dense0/processed/images/").mkdir(parents=True, exist_ok=True)
    image_files = glob.glob(root+'/*.jpg')
    length = len(image_files)
    for i, file_path in enumerate(image_files):
        print('processing [{0}/{1}] path: {2}'.format(i, length, file_path))
        img_ori = np.float32(io.imread(file_path))/255.0
        img = resize(img_ori, (input_height, input_width), order = 1)
        input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
        input_img = input_img.unsqueeze(0)

        input_images = Variable(input_img.to(device) )
        pred_log_depth = model.netG.forward(input_images) 
        pred_log_depth = torch.squeeze(pred_log_depth)

        pred_depth = torch.exp(pred_log_depth)
        pred_inv_depth = pred_depth#1/pred_depth
        pred_inv_depth = pred_inv_depth.data.cpu().numpy()
        
        # you might also use percentile for better visualization
        pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)
        pred_inv_depth = np.clip(pred_inv_depth * 5,0,1)
        file_name = os.path.split(file_path)[-1]
        img_ori_resized = resize(img_ori, (480,640), order = 1)
        pred_inv_depth = resize(pred_inv_depth, (240,320), order = 1)
        io.imsave(root.replace('imgs','processed/images') + file_name, img_ori_resized)
        io.imsave(root.replace('imgs','processed/depth') + file_name.replace('jpg','png'), pred_inv_depth)


test_simple(model)
print("convertion done")
