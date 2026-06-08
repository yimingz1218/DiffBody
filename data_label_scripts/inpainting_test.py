# from diffusers import StableDiffusionInpaintPipeline
# import torch
# import PIL
# from PIL import Image, ImageDraw

# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-inpainting",
#     torch_dtype=torch.float32,
# )
# pipe.to("cuda")
# prompt = "A realistic digital image of a full-body human, extremely detailed, 64K resolution."
# original_image = Image.open('/fsx_laion/alvin/visualization/2.png')
# side_length = max(original_image.width, original_image.height) * 3
# square_image = Image.new("RGB", (side_length, side_length), color="white")
# x_offset = (side_length - original_image.width) // 2
# y_offset = (side_length - original_image.height) // 3
# square_image.paste(original_image, (x_offset, y_offset))

# # Create a mask image with all black pixels
# mask_image = Image.new("L", (side_length, side_length), color=255)

# # Create a rectangular region for the original image and fill it with white
# mask_rect = (x_offset, y_offset, x_offset + original_image.width, y_offset + original_image.height)
# mask_image.paste(0, mask_rect)

# square_image.save('/fsx_laion/alvin/visualization/2_square.png')
# mask_image.save('/fsx_laion/alvin/visualization/2_mask.png')
# # mask_image = 
# #image and mask_image should be PIL images.
# #The mask structure is white for inpainting and black for keeping as is
# image = pipe(prompt=prompt, image=square_image, mask_image=mask_image).images[0]
# image.save("/fsx_laion/alvin/visualization/2_outpainting.png")

# original_image = Image.open('/fsx_laion/alvin/visualization/31.png')
# side_length = max(original_image.width, original_image.height) * 3
# square_image = Image.new("RGB", (side_length, side_length), color="white")
# x_offset = (side_length - original_image.width) // 2
# y_offset = (side_length - original_image.height) // 3
# square_image.paste(original_image, (x_offset, y_offset))

# # Create a mask image with all black pixels
# mask_image = Image.new("L", (side_length, side_length), color=255)

# # Create a rectangular region for the original image and fill it with white
# mask_rect = (x_offset, y_offset, x_offset + original_image.width, y_offset + original_image.height)
# mask_image.paste(0, mask_rect)

# square_image.save('/fsx_laion/alvin/visualization/31_square.png')
# mask_image.save('/fsx_laion/alvin/visualization/31_mask.png')
# # mask_image = 
# #image and mask_image should be PIL images.
# #The mask structure is white for inpainting and black for keeping as is
# image = pipe(prompt=prompt, image=square_image, mask_image=mask_image).images[0]
# image.save("/fsx_laion/alvin/visualization/31_outpainting.png")

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
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
import time

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

def parse():
    parser = argparse.ArgumentParser(description='PyTorch DDP Training')
    # parser.add_argument('data', metavar='DIR',
    #                     help='path to dataset')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--data-path', default='/fsx/laion/data/openprompts.csv', type=str, metavar='PATH',
                        nargs='+',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--dataset-type', default='csv', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--output-path', default='dummy', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=True,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=True,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument("--cuda_visible_devices", default="-1", type=str)

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling sync BN.')
    args = parser.parse_args()
    return args


def main():

    args = parse()
    
    if args.cuda_visible_devices != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cudnn.benchmark = True
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    openpose_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )
    
    to_tensor = transforms.ToTensor()
    
    identity = lambda x: x

    # @mst: use wds dataset API
    class Args:
        pass  # to use open clip api

    data_args = Args()
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
    data_args.batch_size = args.batch_size
    # data_args.world_size = torch.distributed.get_world_size()
    data_args.workers = args.workers
    data_args.seed = -1
    
    # os.makedirs("/fsx_laion/alvin/Dataset/laion2B-en-aesthetic-4.5plus-512_human_structural", exist_ok=True)
    total_num = success_num = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    
    # from mmpose.apis import MMPoseInferencer
    # import mmcv
    # # body_inferencer = MMPoseInferencer("wholebody")
    # body_inferencer = MMPoseInferencer(pose2d='/fsx_laion/alvin/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
    #                                    pose2d_weights='/fsx_laion/alvin/pretrain/ViTPose/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
    #                                    det_model='/fsx_laion/alvin/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
    #                                    det_weights="/fsx_laion/alvin/pretrain/ViTPose/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")
    # face_inferencer = MMPoseInferencer('face')
    # hand_inferencer = MMPoseInferencer('hand')
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "/fsx_laion/alvin/pretrain/stable-diffusion-2-inpainting",
        # torch_dtype=torch.float64,
    )
    pipe.to("cuda")
    
    # sink_writers = {}
    # for name in data_args.train_data:
        # sink_writers[name.split('/')[-1]] = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", name.split('/')[-1]))
    # os.makedirs("/fsx_laion/alvin/yolo_filter", exist_ok=True)
    os.makedirs('/fsx_laion/alvin/visualization/inpainting/', exist_ok=True)
    cnt = 0
    generator = torch.Generator(device=pipe.device).manual_seed(0)
    # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", data_args.train_data[0].split('/')[-1]))
    for tar_file in tqdm(all_tar_list):
        # try:
        data_args.train_data = [tar_file]
        wds_dataset = get_wds_dataset_filter(data_args, identity)
        train_dataloader = wds_dataset.dataloader
        # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/laion2B-en-aesthetic-4.5plus-512_human_structural", tar_file.split('/')[-1]))
        with torch.no_grad():
            for mini_batch, batch in enumerate(train_dataloader):
                # try:
                original, images, text, key, url, json = batch
                
                for i, image in enumerate(images):       
                    # prompt = "A realistic digital image of a human, extremely detailed, 64K resolution."
                    # prompt = text[i].decode('utf-8') + " A realistic digital image of a full-body human, extremely detailed, 64K resolution."
                    prompt = text[i].decode('utf-8') + " 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3."
                    # print(prompt)
                    # print(text.decode('utf-8'))
                    # exit(0)
                    original_image = image
                    if original_image.width >= original_image.height:
                        side_length = int(original_image.width * 1.6)
                    else:
                        side_length = int(original_image.height * 1.2)
                    square_image = Image.new("RGB", (side_length, side_length), color="white")
                    x_offset = (side_length - original_image.width) // 2
                    y_offset = (side_length - original_image.height) // 4
                    square_image.paste(original_image, (x_offset, y_offset))

                    # Create a mask image with all black pixels
                    mask_image = Image.new("L", (side_length, side_length), color=255)

                    # Create a rectangular region for the original image and fill it with white
                    mask_rect = (x_offset, y_offset, x_offset + original_image.width, y_offset + original_image.height)
                    mask_image.paste(0, mask_rect)

                    # square_image.save('/fsx_laion/alvin/visualization/inpainting/{}_square.png')
                    # mask_image.save('/fsx_laion/alvin/visualization/2_mask.png')
                    # mask_image = 
                    #image and mask_image should be PIL images.
                    #The mask structure is white for inpainting and black for keeping as is
                    inpaint_image = pipe(prompt=prompt, 
                                         height=side_length,
                                         width=side_length,
                                         image=square_image, 
                                         mask_image=mask_image, 
                                         num_inference_steps=50,
                                         generator=generator,
                                         negative_prompt="frame, computer, phone, cellphone, monitor, screen, magazine, cover, ads, \
                                                picture in picture, \
                                                multiple humans, human occluded by objects, \
                                                image collections, album, collage, grid, gallery, slide show, \
                                                (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
                                        ).images[0]
                    inpaint_image.paste(original_image, (x_offset, y_offset))
                    inpaint_image.resize((512, 512))
                    square_image.resize((512, 512))
                    square_image.save(f'/fsx_laion/alvin/visualization/inpainting/{mini_batch * args.batch_size + i}_square.png')
                    inpaint_image.save(f'/fsx_laion/alvin/visualization/inpainting/{mini_batch * args.batch_size + i}.png')
                    cnt += 1
                    if cnt >= 50:
                        exit(0)

if __name__ == '__main__':
    main()
