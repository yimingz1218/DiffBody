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
from diffusers import StableDiffusionPipeline
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
import pytorch_lightning as pl

import torch.nn.functional as F
import clip
from torchvision import transforms as T
import timm

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

watermark_preprocessing = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


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
    # numbers = [333, 334, 335, 355, 383, 385, 390, 397, 435, 436, 444, 463, 590, 595, 1243, 1244, 1259, 12929, 13029, 13179, 13279, 13379, 13479, 13529, 13729, 13979, 14029, 14329]
    # all_tar_list = [all_tar_list[x] for x in numbers[args.start : args.end]]
    # all_tar_list = [all_tar_list[x] for x in numbers]

    print(f'Found {len(all_tar_list)} .tar files in {args.data_path}')
    
    # os.makedirs("/fsx_laion/alvin/Dataset/laion2B-en-aesthetic-4.5plus-512_human_structural", exist_ok=True)
    total_num = success_num = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

    s = torch.load("/fsx_laion/alvin/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo

    model.load_state_dict(s)

    model.to("cuda")
    model.eval()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load("/fsx_laion/alvin/improved-aesthetic-predictor/ViT-L-14.pt", device=device)  #RN50x64
    
    watermark_model = timm.create_model(
        'efficientnet_b3a', pretrained=True, num_classes=2)
    
    # checkpoint = torch.load(model_checkpoint_path)
    # watermark_model.load_state_dict(checkpoint['state_dict'])

    watermark_model.classifier = nn.Sequential(
        # 1536 is the orginal in_features
        nn.Linear(in_features=1536, out_features=625),
        nn.ReLU(),  # ReLu to be the activation function
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2),
    )

    watermark_state_dict = torch.load('/fsx_laion/alvin/LAION-5B-WatermarkDetection/models/watermark_model_v1.pt')

    watermark_model.load_state_dict(watermark_state_dict)
    watermark_model.eval()

    if torch.cuda.is_available():
        watermark_model.cuda()
    
    # sink_writers = {}
    # for name in data_args.train_data:
        # sink_writers[name.split('/')[-1]] = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", name.split('/')[-1]))
    os.makedirs("/fsx_laion/alvin/Dataset/getty_human_aes", exist_ok=True)
    # os.makedirs('/fsx_laion/alvin/visualization/mmpose/', exist_ok=True)
    cnt = 0
    fail_list = []
    # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", data_args.train_data[0].split('/')[-1]))
    for tar_file in tqdm(all_tar_list):
        try:
            tar_name = tar_file.split('/')[-1]
            os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/getty_human_aes', tar_name)}")
            writer = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_human_aes", tar_name))
            dataset = wds.WebDataset(tar_file)
            # dataset_inpaint = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/getty_human_inpaint", tar_name))
            # dataset_ldmk = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/getty_human_ldmk", tar_name))
            sample_count = 0
            for sample in dataset:
                modified_sample = dict(sample)  # Create a copy of the sample
                # modified_sample = {}
                # modified_sample["__key__"] = sample["__key__"]
                # modified_sample["__url__"] = sample["__url__"]
                # modified_sample["inpaint"] = sample2["inpaint"]
                # modified_sample["location"] = sample2["location"]
                # modified_sample["ldmk"] = sample1["ldmk"]
                # print(sample["__key__"])
                
                with io.BytesIO(sample["jpg"]) as stream:
                    try:
                        img = PIL.Image.open(stream)
                        img.load()
                        img = img.convert("RGB")
                    except:
                        print("A broken image is encountered, skip")
                        continue
                    
                image = preprocess(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    image_features = model2.encode_image(image)

                    im_emb_arr = normalized(image_features.cpu().detach().numpy())

                    prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)).detach().cpu().numpy()
                    
                modified_sample['aesthetic_score_laion_v2'] = prediction.astype(np.float32).tobytes()
                
                watermark_im = watermark_preprocessing(img).cuda()
                # batch = torch.stack([watermark_im, watermark_im])
                batch = watermark_im.unsqueeze(0)
                with torch.no_grad():
                    pred = watermark_model(batch)
                    syms = F.softmax(pred, dim=1)[0][0].detach().cpu().numpy()
                modified_sample['watermark_score'] = syms.astype(np.float32).tobytes()
                               
                # Write the modified sample back to the tar file
                writer.write(modified_sample)
                # sample_count += 1
                # if sample_count >= 10:
                #     break

            # print("Number of samples in dataset:", sample_count)
            writer.close()
        except:
            os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/getty_human_aes', tar_file.split('/')[-1])}")
            fail_list.append(tar_file.split('/')[-1])
            
    end = time.time()
    
    # print(f"Process {len(all_tar_list)} .tar files, totally contain {total_num} images.")  
    # max_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)  # Convert to megabytes
    # print(f"Peak GPU memory usage: {max_memory:.2f} MB")
    # print(f"The batch size is {args.batch_size}")
    print(f"The overall time cost is {end - start} seconds, avg each tar file process time is {(end - start) / len(all_tar_list)} seconds")
    print(fail_list)


if __name__ == '__main__':
    main()
