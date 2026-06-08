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

import paddle

sys.path.insert(0, '../PaddleSeg')
from paddleseg.cvlibs import manager, Config, SegBuilder
from paddleseg.utils import get_sys_env, logger, get_image_list, utils
from paddleseg.core import predict
from paddleseg.transforms import Compose

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
    all_tar_list = all_tar_list[args.start : args.end]

    print(f'Found {len(all_tar_list)} .tar files in {args.data_path}')
    
    # os.makedirs("/fsx_laion/alvin/Dataset/laion2B-en-aesthetic-4.5plus-512_human_structural", exist_ok=True)
    total_num = success_num = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    
    args["config"] = '/fsx_laion/alvin/PaddleSeg/contrib/PP-HumanSeg/configs/human_pp_humansegv1_server.yml'
    args["model_path"] = "/fsx_laion/alvin/PaddleSeg/contrib/PP-HumanSeg/pretrained_models/human_pp_humansegv1_server_512x512_pretrained/model.pdparams"
    
    os.makedirs('/fsx_laion/alvin/Dataset/getty_human_dino', exist_ok=True)
    cnt = 0
    fail_list = []
    for tar_file in tqdm(all_tar_list):
        # try:
        writer = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_human_dino", tar_file.split('/')[-1]))
        dataset = wds.WebDataset(tar_file)
        sample_count = 0
        for i, sample in enumerate(dataset):
            modified_sample = dict(sample)  # Create a copy of the sample
            
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
            args["vertical_screen"] = original_height > original_width
            with torch.no_grad():
                # fg_pca = PCA(n_components=1)
                # object_pca = PCA(n_components=3)
                print(original_height, original_width)
                img = img_transform(img).unsqueeze(0).to(device)
                print(img.shape)
                test_result = model.forward_features(img)
                print(test_result.keys())
                print(test_result['x_norm_clstoken'].shape)
                print(test_result['x_norm_patchtokens'].shape)
                print(test_result['x_prenorm'].shape)
                print(test_result['masks'].shape)
                exit(0)
                # test_patch_tokens = test_result['x_norm_patchtokens'].detach().cpu().numpy().reshape([2304, -1])
                # fg_result = fg_pca.transform(test_patch_tokens)
                # fg_result = minmax_scale(fg_result)

                # fg_mask = (fg_result > 0.5)

                # object_result = object_pca.transform(test_patch_tokens)
                # object_result = minmax_scale(object_result)

                # only_object = np.zeros_like(object_result)
                # only_object[fg_mask.ravel(), :] = object_result[fg_mask.ravel(), :]
                # print(only_object.shape)
                # exit(0)
                trans_back = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((original_height, original_width), interpolation=PIL.Image.BILINEAR),
                                ])
                normal_pil = trans_back(normal_output[0])
                image_file = io.BytesIO()
                # Convert the image to the desired format and save it to the file-like object
                normal_pil = normal_pil.convert('RGB')  # Convert to RGB mode if necessary
                # normal_pil.save(image_file, format='PNG')  # Replace 'JPEG' with the desired format
                normal_pil.save(image_file, format='JPEG')  # Replace 'JPEG' with the desired format

                # Get the byte string representation of the image
                normal_pil = image_file.getvalue()
                modified_sample["omni_normal"] = normal_pil

            # Add a new attribute to the modified sample
            # modified_sample["new_attribute"] = "new_value"

            # Write the modified sample back to the tar file
            writer.write(modified_sample)
            sample_count += 1

        print("Number of samples in dataset:", sample_count)
        writer.close()
        # except:
        #     os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/getty_human_dino', tar_file.split('/')[-1])}")
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
