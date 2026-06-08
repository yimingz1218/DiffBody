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

def parse():
    parser = argparse.ArgumentParser(description='PyTorch DDP Training')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--data-path', default='/fsx_laion/alvin/Dataset/getty_images_webdataset_human/', type=str, metavar='PATH',
                        nargs='+',
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    return args


def main():

    args = parse()

    all_tar_list = []
    for data_folder in args.data_path:
        all_tar_list += [os.path.join(data_folder, x)
                                 for x in sorted(os.listdir(data_folder)) if
                                 x.endswith('.tar')]  # open('s3_urls.txt').read().splitlines()
    # all_tar_list = all_tar_list[args.start : args.end]
    # all_tar_list = ['/fsx_laion/alvin/Dataset/getty_images_webdataset_human/3a00453a-497c-4bcd-bb70-b7c9ad6d9a4f.tar', '/fsx_laion/alvin/Dataset/getty_images_webdataset_human/072526fe-4576-4e46-be2d-c7bae7e2e246.tar']
    # numbers = [24733, 25334, 25732, 25770, 26008, 26046, 26084, 26522, 26560, 27874, 30763]
    # numbers = [333, 334, 335, 355, 383, 385, 390, 397, 435, 436, 444, 463, 590, 595, 1243, 1244, 1259, 12929, 13029, 13179, 13279, 13379, 13479, 13529, 13729, 13979, 14029, 14329]
    numbers = [4996, 5446]
    all_tar_list = [all_tar_list[x] for x in numbers]
    # all_tar_list = [all_tar_list[x] for x in numbers[args.start : args.end]]
    print(f'Found {len(all_tar_list)} .tar files in {args.data_path}')

    total_num = success_num = 0
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()

    os.makedirs('/fsx_laion/alvin/Dataset/coyo_human_final', exist_ok=True)
    cnt = 0
    fail_list = []
    for tar_file in all_tar_list:
        # try:
        tar_name = tar_file.split('/')[-1]
        os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/coyo_human_final', tar_name)}")
        writer = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/coyo_human_final", tar_name))
        dataset_ori = wds.WebDataset(tar_file)
        dataset_new = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/coyo_human_newpose", tar_name))
        # sample_count = 0
        for sample1, sample2 in zip(dataset_ori, dataset_new):
            new_sample = dict(sample1)  # Create a copy of the sample
            total_num += 1
            new_sample["inpaint"] = sample2["inpaint"]
            new_sample["location"] = sample2["location"]
            new_sample["ldmk"] = sample2["ldmk"]
            new_sample["new_wholebody_bbox"] = sample2["new_wholebody_bbox"]
            new_sample["new_wholebody_bbox_score"] = sample2["new_wholebody_bbox_score"]
            new_sample["new_wholebody_kp"] = sample2["new_wholebody_kp"]
            new_sample["new_wholebody_kp_score"] = sample2["new_wholebody_kp_score"]
            new_sample["new_i_wholebody_bbox"] = sample2["new_i_wholebody_bbox"]
            new_sample["new_i_wholebody_bbox_score"] = sample2["new_i_wholebody_bbox_score"]
            new_sample["new_i_wholebody_kp"] = sample2["new_i_wholebody_kp"]
            new_sample["new_i_wholebody_kp_score"] = sample2["new_i_wholebody_kp_score"]
            new_sample["new_body_bbox"] = sample2["new_body_bbox"]
            new_sample["new_body_bbox_score"] = sample2["new_body_bbox_score"]
            new_sample["new_body_kp"] = sample2["new_body_kp"]
            new_sample["new_body_kp_score"] = sample2["new_body_kp_score"]
            new_sample["new_i_body_bbox"] = sample2["new_i_body_bbox"]
            new_sample["new_i_body_bbox_score"] = sample2["new_i_body_bbox_score"]
            new_sample["new_i_body_kp"] = sample2["new_i_body_kp"]
            new_sample["new_i_body_kp_score"] = sample2["new_i_body_kp_score"]
            new_sample["new_face_bbox"] = sample2["new_face_bbox"]
            new_sample["new_face_bbox_score"] = sample2["new_face_bbox_score"]
            new_sample["new_face_kp"] = sample2["new_face_kp"]
            new_sample["new_face_kp_score"] = sample2["new_face_kp_score"]
            new_sample["new_i_face_bbox"] = sample2["new_i_face_bbox"]
            new_sample["new_i_face_bbox_score"] = sample2["new_i_face_bbox_score"]
            new_sample["new_i_face_kp"] = sample2["new_i_face_kp"]
            new_sample["new_i_face_kp_score"] = sample2["new_i_face_kp_score"]
            new_sample["new_hand_bbox"] = sample2["new_hand_bbox"]
            new_sample["new_hand_bbox_score"] = sample2["new_hand_bbox_score"]
            new_sample["new_hand_kp"] = sample2["new_hand_kp"]
            new_sample["new_hand_kp_score"] = sample2["new_hand_kp_score"]
            new_sample["new_i_hand_bbox"] = sample2["new_i_hand_bbox"]
            new_sample["new_i_hand_bbox_score"] = sample2["new_i_hand_bbox_score"]
            new_sample["new_i_hand_kp"] = sample2["new_i_hand_kp"]
            new_sample["new_i_hand_kp_score"] = sample2["new_i_hand_kp_score"]

            writer.write(new_sample)
            # sample_count += 1

        # print("Number of samples in dataset:", sample_count)
        writer.close()
        # except:
        #     os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/coyo_human_final', tar_file.split('/')[-1])}")
        #     fail_list.append(tar_file.split('/')[-1])

    end = time.time()
    
    # print(f"Process {len(all_tar_list)} .tar files, totally contain {total_num} images.")  
    # print(f"The overall time cost is {end - start} seconds, avg each tar file process time is {(end - start) / len(all_tar_list)} seconds")
    print(fail_list)

if __name__ == '__main__':
    main()
