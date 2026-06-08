import os
import argparse
import torch
import torch.nn.functional as F
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
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
import time
import matplotlib.pyplot as plt

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
import copy

import face_alignment

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
    # all_tar_list = all_tar_list[args.start : args.end]
    # numbers = []
    # with open("/fsx_laion/alvin/sd4human/ldmk_laion-miss2.txt", "r") as file:
    #     for line in file:
    #         numbers.append(int(line.strip()))
    numbers = [3131, 3813, 6688, 7252, 13302, 15346, 16000, 18655, 18760, 21798, 22593, 24478, 28333, 31806, 32878, 36923, 37105, 37131, 37316, 40269, 41137, 48132, 50331, 54317, 54689, 62343, 63900, 64188, 65160, 66175, 66429, 74706, 75233, 75338, 75394, 75741, 77303, 78201, 79560, 87249, 94280, 94864, 100547, 102975, 104178, 106280, 113181, 121543, 124164, 129458, 135667]
    all_tar_list = [all_tar_list[x] for x in numbers[args.start : args.end]]

    print(f'Found {len(all_tar_list)} .tar files in {args.data_path}')
    
    # os.makedirs("/fsx_laion/alvin/Dataset/laion2B-en-aesthetic-4.5plus-512_human_structural", exist_ok=True)
    total_num = success_num = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=True, device="cuda")
    
    os.makedirs('/fsx_laion/alvin/Dataset/laion_human_ldmk', exist_ok=True)
    cnt = 0
    fail_list = []
    for tar_file in tqdm(all_tar_list):
        # try:
        os.system(f"sudo rm -f {os.path.join('/fsx_laion/alvin/Dataset/laion_human_ldmk', tar_file.split('/')[-1])}")
        writer = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/laion_human_ldmk", tar_file.split('/')[-1]))
        dataset = wds.WebDataset(tar_file)
        sample_count = 0
        for i, sample in enumerate(dataset):
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

            original_height = img.height
            original_width = img.width 
            with torch.no_grad():
                try:
                    preds = fa.get_landmarks(np.array(img))
                    lands = []
                    if preds is not None:
                        for pred in preds:
                            land = pred.reshape(-1, 3)[:,:3]
                            lands.append(land)
                except:
                    lands = []
                        
                if len(lands) > 0:        
                    lands = np.stack(lands)
                    modified_sample["ldmk"] = lands.astype(np.float32).tobytes()
                else:
                    modified_sample["ldmk"] = np.array(lands).astype(np.float32).tobytes()

            # Write the modified sample back to the tar file
            writer.write(modified_sample)
            sample_count += 1

        print("Number of samples in dataset:", sample_count)
        writer.close()
        # except:
        #     os.system(f"sudo rm -f {os.path.join('/fsx_laion/alvin/Dataset/laion_human_ldmk', tar_file.split('/')[-1])}")
        #     fail_list.append(tar_file.split('/')[-1])

    end = time.time()
    
    print(f"Process {len(all_tar_list)} .tar files, totally contain {total_num} images.")  
    max_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)  # Convert to megabytes
    print(f"Peak GPU memory usage: {max_memory:.2f} MB")
    # print(f"The batch size is {args.batch_size}")
    print(f"The overall time cost is {end - start} seconds, avg each tar file process time is {(end - start) / len(all_tar_list)} seconds")
    print(fail_list)

if __name__ == '__main__':
    main()
