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
from diffusers import StableDiffusionPipeline, StableDiffusionDepth2ImgPipeline
from tqdm import tqdm
import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
import time
from transformers import CLIPTextModel, CLIPTokenizer, DPTFeatureExtractor, DPTForDepthEstimation

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
import contextlib

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
    # data_args.train_data = ['/fsx_laion/coyo_images_webdataset/00016fdf-75da-48f1-8849-7b7500ddc4f9.tar']
    for data_folder in args.data_path:
        all_tar_list += [os.path.join(data_folder, x)
                                 for x in sorted(os.listdir(data_folder)) if
                                 x.endswith('.tar')]  # open('s3_urls.txt').read().splitlines()
    all_tar_list = all_tar_list[args.start : args.end]
    # numbers = [333, 334, 335, 355, 383, 385, 390, 397, 435, 436, 444, 463, 590, 595, 1243, 1244, 1259, 12929, 13029, 13179, 13279, 13379, 13479, 13529, 13729, 13979, 14029, 14329]
    # all_tar_list = [all_tar_list[x] for x in numbers[args.start : args.end]]
    # all_tar_list = [all_tar_list[x] for x in numbers]

    print(f'Found {len(all_tar_list)} .tar files in {args.data_path}')
    
    # os.makedirs("/fsx_laion/alvin/Dataset/laion2B-en-aesthetic-4.5plus-512_human_structural", exist_ok=True)
    total_num = success_num = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    
    feature_extractor = DPTFeatureExtractor.from_pretrained("/fsx_laion/alvin/pretrain/stable-diffusion-2-depth/feature_extractor")
    depth_estimator = DPTForDepthEstimation.from_pretrained("/fsx_laion/alvin/pretrain/stable-diffusion-2-depth", subfolder="depth_estimator").to(device)
    
    # sink_writers = {}
    # for name in data_args.train_data:
        # sink_writers[name.split('/')[-1]] = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/coyo_images_webdataset_human", name.split('/')[-1]))
    os.makedirs("/fsx_laion/alvin/Dataset/coyo_human_midas", exist_ok=True)
    # os.makedirs('/fsx_laion/alvin/visualization/mmpose/', exist_ok=True)
    cnt = 0
    fail_list = []
    # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/coyo_images_webdataset_human", data_args.train_data[0].split('/')[-1]))
    for tar_file in tqdm(all_tar_list):
        # try:
        tar_name = tar_file.split('/')[-1]
        os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/coyo_human_midas', tar_name)}")
        writer = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/coyo_human_midas", tar_name))
        dataset = wds.WebDataset(tar_file)
        # dataset_inpaint = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/coyo_human_inpaint", tar_name))
        # dataset_ldmk = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/coyo_human_ldmk", tar_name))
        sample_count = 0
        for sample in dataset:
            modified_sample = dict(sample)  # Create a copy of the sample
            
            with io.BytesIO(sample["jpg"]) as stream:
                try:
                    img = PIL.Image.open(stream)
                    img.load()
                    img = img.convert("RGB")
                except:
                    print("A broken image is encountered, skip")
                    continue
                
            with torch.no_grad():
                pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device=device)
                # The DPT-Hybrid model uses batch-norm layers which are not compatible with fp16.
                # So we use `torch.autocast` here for half precision inference.
                context_manger = torch.autocast("cuda", dtype=torch.float32) if device.type == "cuda" else contextlib.nullcontext()
                with context_manger:
                    depth_map = depth_estimator(pixel_values).predicted_depth
                # depth_map = torch.nn.functional.interpolate(
                #     depth_map.unsqueeze(0),
                #     size=(img.height, img.width),
                #     mode="bicubic",
                #     align_corners=False,
                # )
                depth_min = torch.amin(depth_map, dim=[0, 1, 2], keepdim=True)
                depth_max = torch.amax(depth_map, dim=[0, 1, 2], keepdim=True)
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
                depth_map = depth_map.squeeze(0)
                # # print(depth_map.shape)
                numpy_image = depth_map.cpu().numpy()

                # Convert the NumPy array to a PIL image
                pil_image = Image.fromarray(np.uint8(numpy_image * 255))
                image_file = io.BytesIO()
                # Convert the image to the desired format and save it to the file-like object
                depth_pil = pil_image.convert('RGB')  # Convert to RGB mode if necessary
                # normal_pil.save(image_file, format='PNG')  # Replace 'JPEG' with the desired format
                depth_pil.save(image_file, format='JPEG')  # Replace 'JPEG' with the desired format

                # Get the byte string representation of the image
                depth_pil = image_file.getvalue()
                modified_sample["midas_depth"] = depth_pil
                # img.save('/fsx_laion/alvin/visualization/midas/img.png')
                # pil_image.save('/fsx_laion/alvin/visualization/midas/depth.png')
                # exit(0)
                # Add a new attribute to the modified sample
                # modified_sample["new_attribute"] = "new_value"
                
                # Write the modified sample back to the tar file
            writer.write(modified_sample)
                # sample_count += 1
                # if sample_count >= 10:
                #     break

        # print("Number of samples in dataset:", sample_count)
        writer.close()
        # except:
        #     os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/coyo_human_midas', tar_file.split('/')[-1])}")
        #     fail_list.append(tar_file.split('/')[-1])

    end = time.time()
    
    # print(f"Process {len(all_tar_list)} .tar files, totally contain {total_num} images.")  
    # max_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)  # Convert to megabytes
    # print(f"Peak GPU memory usage: {max_memory:.2f} MB")
    # print(f"The batch size is {args.batch_size}")
    print(f"The overall time cost is {end - start} seconds, avg each tar file process time is {(end - start) / len(all_tar_list)} seconds")
    print(fail_list)

if __name__ == '__main__':
    main()
