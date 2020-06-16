import glob
import os
from pathlib import Path
import cv2

#generate train.csv from reading the images from disk
root = 'megadepth_subset/depth/'
image_files = glob.glob(root+'/*.png')
result = ''
for i, file_path in enumerate(image_files):
    dpath = file_path.replace('megadepth_subset','data')
    ipath = dpath.replace('depth','images').replace('png','jpg')
    result = result + ipath + ','+dpath +'\n'
    print(ipath)
    
text_file = open("megadepth_train.csv", "w")
text_file.write(result)
text_file.close()
