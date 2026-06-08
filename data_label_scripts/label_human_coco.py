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
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

from openclip.training.data import get_wds_dataset, get_wds_dataset_filter

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
from transformers import CLIPTextModel, CLIPTokenizer, DPTFeatureExtractor, DPTForDepthEstimation

def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img

class CustomDataset(Dataset):
    def __init__(self, json_path='/fsx_laion/alvin/Dataset/coco/512x512-img_text_pose_val2014.json', image_processor=None):
        self.image_data = []
        self.text_data = []
        self.id_list = []
        self.image_processor = image_processor

        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
        
        for image_id, caption in json_data.items():
            self.image_data.append(f'/fsx_laion/alvin/Dataset/coco/512x512-val2014/COCO_val2014_{int(image_id):012d}.jpg')
            self.text_data.append(caption)
            self.id_list.append(int(image_id))
            
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        image_path = self.image_data[idx]
        image = Image.open(image_path).convert("RGB")
        
        resize = transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC)
        image = resize(image)
        
        if self.image_processor:
            image = self.image_processor(images=image, return_tensors="pt")

        return image, self.id_list[idx]

def parse():
    parser = argparse.ArgumentParser(description='PyTorch DDP Training')
    
    args = parser.parse_args()
    return args

def collate_fn(batch):
    images, ids = zip(*batch)
    # images = torch.stack(images)
    
    return images, ids

def main():

    args = parse()

    cudnn.benchmark = True

    import sys
    sys.path.insert(0, '../omnidata/omnidata_tools/torch')
    from modules.unet import UNet
    from modules.midas.dpt_depth import DPTDepthModel
    from data.transforms import get_transform
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_size = 384
    normal_pretrained_weights_path = '/fsx_laion/alvin/omnidata/omnidata_tools/torch/pretrained_models/omnidata_dpt_normal_v2.ckpt'
    normal_model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
    normal_checkpoint = torch.load(normal_pretrained_weights_path, map_location=map_location)
    if 'state_dict' in normal_checkpoint:
        normal_state_dict = {}
        for k, v in normal_checkpoint['state_dict'].items():
            normal_state_dict[k[6:]] = v
    else:
        normal_state_dict = normal_checkpoint

    normal_model.load_state_dict(normal_state_dict)
    normal_model.to(device)
    normal_trans_totensor = transforms.Compose([
                                transforms.Resize((image_size, image_size), interpolation=PIL.Image.BILINEAR),
                                # transforms.CenterCrop(image_size),
                                get_transform('rgb', image_size=None)])
    
    depth_pretrained_weights_path = '/fsx_laion/alvin/omnidata/omnidata_tools/torch/pretrained_models/omnidata_dpt_depth_v2.ckpt'
    # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
    depth_model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
    depth_checkpoint = torch.load(depth_pretrained_weights_path, map_location=map_location)
    if 'state_dict' in depth_checkpoint:
        depth_state_dict = {}
        for k, v in depth_checkpoint['state_dict'].items():
            depth_state_dict[k[6:]] = v
    else:
        depth_state_dict = depth_checkpoint
    depth_model.load_state_dict(depth_state_dict)
    depth_model.to(device)
    depth_trans_totensor = transforms.Compose([
                                transforms.Resize((image_size, image_size), interpolation=PIL.Image.BILINEAR),
                                # transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=0.5, std=0.5)])
    
    feature_extractor = DPTFeatureExtractor.from_pretrained("/fsx_laion/alvin/pretrain/stable-diffusion-2-depth/feature_extractor")
    depth_estimator = DPTForDepthEstimation.from_pretrained("/fsx_laion/alvin/pretrain/stable-diffusion-2-depth", subfolder="depth_estimator").to(device)
    
    dataset = CustomDataset()

    # Create a DataLoader for batching and shuffling
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn)
    
    os.makedirs('/fsx_laion/alvin/Dataset/coco/512x512-val2014-structure/normal', exist_ok=True)
    os.makedirs('/fsx_laion/alvin/Dataset/coco/512x512-val2014-structure/depth', exist_ok=True)
    os.makedirs('/fsx_laion/alvin/Dataset/coco/512x512-val2014-structure/midas_depth', exist_ok=True)

    # Now you can iterate through the dataloader to get batches of data
    for iter, batch in enumerate(tqdm(dataloader)):
        images, id = batch
        img = images[0]
        id = id[0]

        # images = images.to(device)
        with torch.no_grad():
            normal_img_tensor = normal_trans_totensor(img)[:3].unsqueeze(0).to(device)
            if normal_img_tensor.shape[1] == 1:
                normal_img_tensor = normal_img_tensor.repeat_interleave(3,1)

            normal_output = normal_model(normal_img_tensor).clamp(min=0, max=1)
            trans_back = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((512, 512), interpolation=PIL.Image.BILINEAR),
                            ])
            normal_pil = trans_back(normal_output[0])
            normal_pil.save(os.path.join('/fsx_laion/alvin/Dataset/coco/512x512-val2014-structure/normal', f'COCO_val2014_{int(id):012d}.jpg'))
            
            depth_img_tensor = depth_trans_totensor(img)[:3].unsqueeze(0).to(device)
            if depth_img_tensor.shape[1] == 1:
                depth_img_tensor = depth_img_tensor.repeat_interleave(3,1)

            depth_output = depth_model(depth_img_tensor).clamp(min=0, max=1)
            
            depth_output = F.interpolate(depth_output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
            depth_output = depth_output.clamp(0, 1)
            depth_output = 1 - depth_output

            plt.imsave(os.path.join('/fsx_laion/alvin/Dataset/coco/512x512-val2014-structure/depth', f'COCO_val2014_{int(id):012d}.jpg'), depth_output.detach().cpu().squeeze(),cmap='viridis')
            
            
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
            pil_image.save(os.path.join('/fsx_laion/alvin/Dataset/coco/512x512-val2014-structure/midas_depth', f'COCO_val2014_{int(id):012d}.jpg'))
        
if __name__ == '__main__':
    main()
