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
    # data_args.train_data = ['/fsx_laion/laion_images_webdataset/00016fdf-75da-48f1-8849-7b7500ddc4f9.tar']
    for data_folder in args.data_path:
        all_tar_list += [os.path.join(data_folder, x)
                                 for x in sorted(os.listdir(data_folder)) if
                                 x.endswith('.tar')]  # open('s3_urls.txt').read().splitlines()
    all_tar_list = all_tar_list[args.start : args.end]
    # numbers = [8955, 8956, 8957, 8958, 8959, 8960, 8961, 8962, 8963, 8964, 8965, 8966, 8967, 8968, 8969, 8970, 8971, 8972, 8973, 8974, 8975, 8976, 8977, 8978, 8979, 8980, 8981, 8982, 8983, 8984, 8985, 8986, 8987, 8988, 8989, 8990, 8991, 8992, 8993, 8994, 8995, 8996, 8997, 8998, 8999, 26251, 26252, 26253, 26254, 26255, 26256, 26257, 26258, 26259, 26260, 26261, 26262, 26263, 26264, 26265, 26266, 26267, 26268, 26269, 26270, 26271, 26272, 26273, 26274, 26275, 26276, 26277, 26278, 26279, 26280, 26281, 26282, 26283, 26284, 26285, 26286, 26287, 26288, 26289, 26290, 26291, 26292, 26293, 26294, 26295, 26296, 26297, 26298, 26299, 32113, 32114, 32115, 32116, 32117, 32118, 32119, 32120, 32121, 32122, 32123, 32124, 32125, 32126, 32127, 32128, 32129, 32130, 32131, 32132, 32133, 32134, 32135, 32136, 32137, 32138, 32139, 32140, 32141, 32142, 32143, 32144, 32145, 32146, 32147, 32148, 32149]
    # all_tar_list = [all_tar_list[x] for x in numbers[args.start : args.end]]

    print(f'Found {len(all_tar_list)} .tar files in {args.data_path}')
    
    # os.makedirs("/fsx_laion/alvin/Dataset/laion2B-en-aesthetic-4.5plus-512_human_structural", exist_ok=True)
    # total_num = success_num = 0
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    
    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     "/fsx_laion/alvin/pretrain/stable-diffusion-2-inpainting",
    #     # torch_dtype=torch.float64,
    # )
    # pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
    # pipe.to("cuda")
    
    # os.makedirs('/fsx_laion/alvin/visualization/inpainting/', exist_ok=True)
    os.makedirs('/fsx_laion/alvin/Dataset/coyo_human_1024x1024', exist_ok=True)
    # cnt1 = 0
    # cnt2 = 0
    # cnt = 0
    # generator = torch.Generator(device=pipe.device).manual_seed(0)
    
    fail_list = []
    # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/laion_images_webdataset_human", data_args.train_data[0].split('/')[-1]))
    for tar_file in tqdm(all_tar_list):
        # try:
        cnt = 0
        os.system(f"sudo rm -f {os.path.join('/fsx_laion/alvin/Dataset/coyo_human_1024x1024', tar_file.split('/')[-1])}")
        writer = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/coyo_human_1024x1024", tar_file.split('/')[-1]))
        # writer = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/laion_human_posear0.5", '{:06d}.tar'.format(cnt1)))
        dataset = wds.WebDataset(tar_file)

        for i, sample in enumerate(dataset):

            # modified_sample = dict(sample)  # Create a copy of the sample
            # modified_sample = {}
            # modified_sample["__key__"] = sample["__key__"]
            # modified_sample["__url__"] = sample["__url__"]
            
            # with io.BytesIO(sample["jpg"]) as stream:
            #     try:
            #         img = PIL.Image.open(stream)
            #         img.load()
            #         img = img.convert("RGB")
            #     except:
            #         print("A broken image is encountered, skip")
            #         continue
                
            # dict_json = sample["json"]
            string_json = sample["json"].decode('utf-8')
            dict_json = json.loads(string_json)
            # if "height" in dict_json.keys() and "width" in dict_json.keys():
            min_length = min(dict_json["height"], dict_json["width"])
            if min_length >= 1024:
                modified_sample = dict(sample)
                # cnt2 += 1
                writer.write(modified_sample)
                cnt += 1

            # if height >= 1024 and width >= 1024:
            #     modified_sample = dict(sample)
            #     # cnt2 += 1
            #     writer.write(modified_sample)
            #     cnt += 1
        # if cnt2 >= 500:          
        writer.close()
            # cnt2 = 0
            # cnt1 += 1
            
        if cnt == 0:
            os.system(f"sudo rm -f {os.path.join('/fsx_laion/alvin/Dataset/coyo_human_1024x1024', tar_file.split('/')[-1])}")
        
        # except:
        #     os.system(f"sudo rm -f {os.path.join('/fsx_laion/alvin/Dataset/laion_human_1024x1024', tar_file.split('/')[-1])}")
        #     fail_list.append(tar_file.split('/')[-1])
            
    end = time.time()
    # print(cnt)
    # print(f"Process {len(all_tar_list)} .tar files, totally contain {total_num} images.")  
    # max_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)  # Convert to megabytes
    # print(f"Peak GPU memory usage: {max_memory:.2f} MB")
    # # print(f"The batch size is {args.batch_size}")
    # print(f"The overall time cost is {end - start} seconds, avg each tar file process time is {(end - start) / len(all_tar_list)} seconds")
    print(fail_list)

if __name__ == '__main__':
    main()
