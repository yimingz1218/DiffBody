import os
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import numpy as np
from glob import glob
import webdataset as wds
import sys
sys.path.insert(0, '.')
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from diffusers.schedulers import DPMSolverSinglestepScheduler
from tqdm import tqdm
import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
import time

from openclip.training.data import get_wds_dataset, get_wds_dataset_filter, tarfile_to_samples_nothrow

import csv
import json
from pprint import pprint
import pandas as pd
import logging
from PIL import Image
from collections import OrderedDict, defaultdict
import linecache
import random

from typing import Optional, Union
from functools import partial

def parse():
    parser = argparse.ArgumentParser(description='PyTorch DDP Training')
    parser.add_argument('--data-path', default='/fsx/laion/data/openprompts.csv', type=str, metavar='PATH',
                        nargs='+',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    args = parser.parse_args()
    return args


def main():

    args = parse()

    cudnn.benchmark = True
    
    all_tar_list = []
    # data_args.train_data = []
    # all_tar_list = [os.path.join(data_folder, x)
    #                 for x in os.listdir(data_folder) if
    #                 x.endswith('.tar')]
    # data_args.train_data = ['/fsx_laion/getty_images_webdataset/00016fdf-75da-48f1-8849-7b7500ddc4f9.tar']
    for data_folder in args.data_path:
        all_tar_list += [os.path.join(data_folder, x)
                                 for x in sorted(os.listdir(data_folder)) if
                                 x.endswith('.tar')]  # open('s3_urls.txt').read().splitlines()
    numbers = [2600, 6600, 56700, 91416, 91417, 91418, 91419, 91420, 91421, 91422, 91423, 91424, 91425, 91426, 91427, 91428, 91429, 91430, 91431, 91432, 91433, 91434, 91435, 91436, 91437, 91438, 91439, 91440, 91441, 91442, 91443, 91444, 91445, 91446, 91447, 91448, 91449, 91450, 91451, 91452, 91453, 91454, 91455, 91456, 91457, 91458, 91459, 91460, 91461, 91462, 91463, 91464, 91465, 91466, 91467, 91468, 91469, 91470, 91471, 91472, 91473, 91474, 91475, 91476, 91477, 91478, 91479, 91480, 91481, 91482, 91483, 91484, 91485, 91486, 91487, 91488, 91489, 91490, 91491, 91492, 91493, 91494, 91495, 91496, 91497, 91498, 91499, 92200, 92217, 92218, 92219, 92220, 92221, 92222, 92223, 92224, 92225, 92226, 92227, 92228, 92229, 92230, 92231, 92232, 92233, 92234, 92235, 92236, 92237, 92238, 92239, 92240, 92241, 92242, 92243, 92244, 92245, 92246, 92247, 92248, 92249, 92250, 92251, 92252, 92253, 92254, 92255, 92256, 92257, 92258, 92259, 92260, 92261, 92262, 92263, 92264, 92265, 92266, 92267, 92268, 92269, 92270, 92271, 92272, 92273, 92274, 92275, 92276, 92277, 92278, 92279, 92280, 92281, 92282, 92283, 92284, 92285, 92286, 92287, 92288, 92289, 92290, 92291, 92292, 92293, 92294, 92295, 92296, 92297, 92298, 92299, 93016, 93017, 93018, 93019, 93020, 93021, 93022, 93023, 93024, 93025, 93026, 93027, 93028, 93029, 93030, 93031, 93032, 93033, 93034, 93035, 93036, 93037, 93038, 93039, 93040, 93041, 93042, 93043, 93044, 93045, 93046, 93047, 93048, 93049, 93050, 93051, 93052, 93053, 93054, 93055, 93056, 93057, 93058, 93059, 93060, 93061, 93062, 93063, 93064, 93065, 93066, 93067, 93068, 93069, 93070, 93071, 93072, 93073, 93074, 93075, 93076, 93077, 93078, 93079, 93080, 93081, 93082, 93083, 93084, 93085, 93086, 93087, 93088, 93089, 93090, 93091, 93092, 93093, 93094, 93095, 93096, 93097, 93098, 93099, 136890]
    # all_tar_list = all_tar_list[args.start : args.end]
    all_tar_list = [all_tar_list[x] for x in numbers[args.start : args.end]]

    print(f'Found {len(all_tar_list)} .tar files in {args.data_path}')
    
    # os.makedirs("/fsx_laion/alvin/Dataset/laion2B-en-aesthetic-4.5plus-512_human_structural", exist_ok=True)
    total_num = success_num = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "/fsx_laion/alvin/pretrain/stable-diffusion-2-inpainting",
        # torch_dtype=torch.float64,
    )
    pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    
    # os.makedirs('/fsx_laion/alvin/visualization/inpainting/', exist_ok=True)
    os.makedirs('/fsx_laion/alvin/Dataset/laion_human_inpaint', exist_ok=True)
    cnt = 0
    generator = torch.Generator(device=pipe.device).manual_seed(0)
    
    fail_list = []
    # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", data_args.train_data[0].split('/')[-1]))
    for tar_file in all_tar_list:
        try:
            os.system(f"sudo rm -f {os.path.join('/fsx_laion/alvin/Dataset/laion_human_inpaint', tar_file.split('/')[-1])}")
            writer = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/laion_human_inpaint", tar_file.split('/')[-1]))
            dataset = wds.WebDataset(tar_file)

            for i, sample in tqdm(enumerate(dataset)):

                # modified_sample = dict(sample)  # Create a copy of the sample
                modified_sample = {}
                modified_sample["__key__"] = sample["__key__"]
                modified_sample["__url__"] = sample["__url__"]
                
                with io.BytesIO(sample["jpg"]) as stream:
                    try:
                        img = PIL.Image.open(stream)
                        img.load()
                        img = img.convert("RGB")
                    except:
                        print("A broken image is encountered, skip")
                        continue
                    
                prompt = sample["txt"].decode('utf-8') + " 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3."
                # print(prompt)
                # print(text.decode('utf-8'))
                # exit(0)
                original_image = img
                if original_image.width >= original_image.height:
                    side_length = int(original_image.width * 1.6) // 8 * 8
                else:
                    side_length = int(original_image.height * 1.4) // 8 * 8
                square_image = Image.new("RGB", (side_length, side_length), color="white")
                x_offset = (side_length - original_image.width) // 2
                y_offset = (side_length - original_image.height) // 4
                square_image.paste(original_image, (x_offset, y_offset))

                # Create a mask image with all black pixels
                mask_image = Image.new("L", (side_length, side_length), color=255)

                # Create a rectangular region for the original image and fill it with white
                mask_rect = (x_offset, y_offset, x_offset + original_image.width, y_offset + original_image.height)
                mask_image.paste(0, mask_rect)
                
                inpaint_image = pipe(prompt=prompt, 
                                    image=square_image, 
                                    mask_image=mask_image, 
                                    num_inference_steps=20,
                                    generator=generator,
                                    negative_prompt="grid, human on the side, frame, computer, tv, television, phone, cellphone, monitor, screen, magazine, cover, ads, \
                                        picture in picture, two humans, three humans, \
                                        multiple humans, human occluded by objects, \
                                        image collections, album, collage, grid, gallery, slide show, \
                                        (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
                                ).images[0]
                inpaint_image.paste(original_image.resize((int(img.width * 512 / side_length), int(img.height * 512 / side_length))), (int(x_offset * 512 / side_length), int(y_offset * 512 / side_length)))
                # inpaint_image.resize((256, 256))
                # inpaint_image.save(f'/fsx_laion/alvin/visualization/inpainting/{i}.png')
                # if i >= 50:
                #     break
                # exit(0)
                image_file = io.BytesIO()
                # Convert the image to the desired format and save it to the file-like object
                inpaint_image = inpaint_image.convert('RGB')  # Convert to RGB mode if necessary
                # normal_pil.save(image_file, format='PNG')  # Replace 'JPEG' with the desired format
                inpaint_image.save(image_file, format='JPEG')  # Replace 'JPEG' with the desired format

                # Get the byte string representation of the image
                inpaint_image = image_file.getvalue()
                modified_sample["inpaint"] = inpaint_image
                modified_sample["location"] = np.array([
                    x_offset * 512. / side_length, \
                    y_offset * 512. / side_length, \
                    img.width * 512. / side_length, \
                    img.height * 512. / side_length, \
                    side_length, \
                    x_offset, \
                    y_offset, \
                    img.width, \
                    img.height
                ]).astype(np.float32).tobytes()

                writer.write(modified_sample)
                
            writer.close()
                # square_image.save(f'/fsx_laion/alvin/visualization/inpainting/{i}_square.png')
                # inpaint_image.save(f'/fsx_laion/alvin/visualization/inpainting/{i}.png')
                # cnt += 1
                # if cnt >= 50:
                #     exit(0)
            
        except:
            os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/laion_human_inpaint', tar_file.split('/')[-1])}")
            fail_list.append(tar_file.split('/')[-1])
            
    end = time.time()
    
    # print(f"Process {len(all_tar_list)} .tar files, totally contain {total_num} images.")  
    # max_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)  # Convert to megabytes
    # print(f"Peak GPU memory usage: {max_memory:.2f} MB")
    # # print(f"The batch size is {args.batch_size}")
    print(f"The overall time cost is {end - start} seconds, avg each tar file process time is {(end - start) / len(all_tar_list)} seconds")
    print(fail_list)

if __name__ == '__main__':
    main()
