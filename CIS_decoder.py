# coding=UTF-8
# Code Author: Xuzhong Yan
# Patched: make JPEG saving robust for RGBA/LA/P modes and add error handling.

import sys, io, os, base64, shutil
from tqdm import tqdm
import numpy as np
from PIL import Image

def _to_jpeg_compatible(im: Image.Image) -> Image.Image:
    """Return a JPEG-compatible PIL image (RGB or L). Flatten or convert as needed."""
    if im.mode == "RGBA":
        # Flatten alpha onto white background (change to (0,0,0) for black)
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    if im.mode in ("P", "LA"):
        return im.convert("RGB")
    if im.mode not in ("RGB", "L"):
        return im.convert("RGB")
    return im

def rle_decoder(set_name, part_num):  # converting rle data to JPG image
    print(' * Converting RLE data to JPG images')
    txt_path = os.path.join('dataset', 'images', set_name, f'{set_name}-{part_num}.txt')
    image_folder = f'{set_name}-{part_num}'
    os.makedirs(image_folder, exist_ok=True)

    with open(txt_path, 'r', encoding='utf-8') as f:
        txt_data = f.read()
        all_image_data = txt_data.split('\n')[:-1]

        for line in tqdm(all_image_data):
            try:
                image_data = line[1:-1].split(',')
                image_name, rle_data = image_data[0][1:-1], image_data[1][1:-1]

                # decode bytes -> numpy -> PIL
                bio = io.BytesIO()
                bio.write(base64.b64decode(rle_data))
                bio.seek(0)
                image_arr = np.array(Image.open(bio))
                image = Image.fromarray(image_arr)

                # ensure JPEG-compatible
                image = _to_jpeg_compatible(image)

                out_path = os.path.join(image_folder, image_name)
                image.save(out_path, format="JPEG")
            except Exception as e:
                print(f"[WARN] Failed to decode/save '{line[:64]}...': {e}")

def cut_and_paste(set_name, part_num):  # moving all files to one folder
    print(f' * Moving JPG images to {set_name} folder')
    source = f'{set_name}-{part_num}'
    destination = set_name
    os.makedirs(destination, exist_ok=True)

    allfiles = os.listdir(source)
    for fname in tqdm(allfiles):
        src_path = os.path.join(source, fname)
        dst_path = os.path.join(destination, fname)
        shutil.move(src_path, dst_path)

    os.rmdir(source)

if __name__ == '__main__':
    # To avoid OOM, please decode train, val & test set separately.
    dataset_dict = {sys.argv[1]: int(sys.argv[2])}
    for set_name in dataset_dict:
        for part_num in range(1, dataset_dict[set_name] + 1):
            print(f'\n decoding {set_name} set - part {part_num}')
            rle_decoder(set_name, part_num)
            cut_and_paste(set_name, part_num)
