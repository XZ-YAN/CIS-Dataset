# coding=UTF-8
# Code Author: Xuzhong Yan

import sys, io, os, glob, base64, json, ast, random, shutil
from tqdm import tqdm
import numpy as np
from PIL import Image

def rle_decoder(set_name, part_num): # converting rle data to JPG image
    print(' * Converting RLE data to JPG images')
    txt_path = 'dataset/images/'+set_name+'/'+set_name+'-'+str(part_num)+'.txt'
    image_folder = set_name+'-'+str(part_num)+'/'
    if not os.path.exists(image_folder): os.mkdir(image_folder)
    
    with open(txt_path) as f:
        txt_data = f.read()
        all_image_data = txt_data.split('\n')[:-1]
    
        for i in tqdm(all_image_data):
            image_data = i[1:-1].split(',')
            image_name, rle_data = image_data[0][1:-1], image_data[1][1:-1]
            f = io.BytesIO()
            f.write(base64.b64decode(rle_data))
            image_arr = np.array(Image.open(f))
            image = Image.fromarray(image_arr)
            image.save(image_folder + '/' + image_name)
    
def cut_and_paste(set_name, part_num): # moving all files to one folder
    print(' * Moving JPG images to '+set_name+' folder')
    source = set_name+'-'+str(part_num)+'/'
    destination = set_name+'/'
    if not os.path.exists(destination):
        os.makedirs(destination)

    allfiles = os.listdir(source)
    for f in tqdm(allfiles):
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f)
        shutil.move(src_path, dst_path)

    os.rmdir(source)
    
if __name__ == '__main__':
    # To avoid OOM, please decode train, val & test set separately.
    dataset_dict = {sys.argv[1]: int(sys.argv[2])}
    for set_name in dataset_dict:
        for part_num in range(dataset_dict[set_name]):
            part_num = part_num + 1
            print('\n decoding '+set_name+' set - part '+str(part_num))
            rle_decoder(set_name, part_num)
            cut_and_paste(set_name, part_num)  
