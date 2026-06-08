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
    parser.add_argument('-b', '--batch-size', default=1, type=int,
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

    from transformers import YolosImageProcessor, YolosForObjectDetection
    
    model = YolosForObjectDetection.from_pretrained('/fsx_laion/alvin/pretrain/yolos-tiny').to('cuda')
    image_processor = YolosImageProcessor.from_pretrained("/fsx_laion/alvin/pretrain/yolos-tiny")
    
    os.makedirs("/fsx_laion/alvin/Dataset/COYO-700M-512-min-image-size200_human", exist_ok=True)
    total_num = success_num = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    # sink_writers = {}
    # for name in data_args.train_data:
        # sink_writers[name.split('/')[-1]] = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", name.split('/')[-1]))
    # os.makedirs("/fsx_laion/alvin/yolo_filter", exist_ok=True)
    # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", data_args.train_data[0].split('/')[-1]))
    for tar_file in tqdm(all_tar_list):
        try:
            data_args.train_data = [tar_file]
            wds_dataset = get_wds_dataset_filter(data_args, identity)
            train_dataloader = wds_dataset.dataloader
            sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/COYO-700M-512-min-image-size200_human", tar_file.split('/')[-1]))
            with torch.no_grad():
                for mini_batch, batch in enumerate(train_dataloader):
                    try:
                        original, images, text, key, url, json = batch
                        # sink = sink_writers[url[0].split('/')[-1]]
                        # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", url[0].split('/')[-1]))
                        # if len(images) != args.batch_size:
                        #     print(len(images))
                        #     print(url)
                        # assert url[0] == url[-1]
                        # continue
                        inputs = image_processor(images=images, return_tensors="pt").to("cuda")
                        outputs = model(**inputs)

                        # model predicts bounding boxes and corresponding COCO classes
                        logits = outputs.logits
                        bboxes = outputs.pred_boxes

                        # print results
                        target_sizes = torch.stack([torch.tensor(images[i].size[::-1]) for i in range(len(images))])
                        results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)
                        for i, result in enumerate(results):
                            total_num += 1
                            person_num = 0
                            area_ratio = 0.
                            scores = result["scores"]
                            labels = result["labels"]
                            boxs = result["boxes"]
                            for score, label, box in zip(scores, labels, boxs):
                                if model.config.id2label[label.item()] == "person":
                                    box = [round(i, 2) for i in box.tolist()]
                                    area_ratio = max(area_ratio, (box[2] - box[0]) * (box[3] - box[1]) / (target_sizes[i][0] * target_sizes[i][1]))
                                    # print(area_ratio.item())
                                    person_num += 1
                                
                            if person_num > 0 and person_num < 4 and area_ratio > 0.1:
                                success_num += 1
                                sink.write({
                                    "jpg": original[i], 
                                    # "text": text[i].decode('utf-8'),
                                    # "__key__": key[i].decode('utf-8'), 
                                    "txt": text[i],
                                    "__key__": key[i], 
                                    "__url__": url[i], 
                                    "json": json[i],
                                })
                                # img_id = mini_batch * args.batch_size + i                   
                                # images[i].save(os.path.join("/fsx_laion/alvin/yolo_filter", f"{img_id}.png"))
                                    # break
                            # if person_num > 0: print(person_num)
                            #     print(1)
                    except:
                        pass
                sink.close()
        except:
            os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/COYO-700M-512-min-image-size200_human', tar_file.split('/')[-1])}")
            print(f"{tar_file.split('/')[-1]} fails")
            
    # for writer in sink_writers.values():
    #     writer.close()
    end = time.time()
    
    print(f"Process {len(all_tar_list)} .tar files, totally contain {total_num} images, and {success_num} of them contains person, ratio is {success_num * 1.0 / total_num}.")  
    max_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)  # Convert to megabytes
    print(f"Peak GPU memory usage: {max_memory:.2f} MB")
    print(f"The batch size is {args.batch_size}")
    print(f"The overall time cost is {end - start} seconds, avg each tar file process time is {(end - start) / len(all_tar_list)} seconds")
                
            # print(results["labels"].shape)
            # print(results["scores"].shape)
            # print(results["boxes"].shape)
            # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            #     box = [round(i, 2) for i in box.tolist()]
            #     print(
            #         f"Detected {model.config.id2label[label.item()]} with confidence "
            #         f"{round(score.item(), 3)} at location {box}"
            #     )
        
    #     # input_image = resize_image(input_image, detect_resolution)
    #     for i, each in enumerate(images):
    #         input_image = np.array(each, dtype=np.uint8)
    #         input_image = HWC3(input_image)
    #         input_image = resize_image(input_image, resolution=args.resolution)
    #         H, W, C = input_image.shape
    #         results = openpose.detect_poses(input_image, include_hand=True, include_face=True)
            
    #         if len(results) != 0:
    #             # print(len(results))
    #             canvas = draw_poses(results, H, W, draw_body=True, draw_hand=True, draw_face=True) 

    #             detected_map = canvas
    #             detected_map = HWC3(detected_map)
                
    #             img = resize_image(input_image, resolution=args.resolution)
    #             H, W, C = img.shape

    #             detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    #             detected_map = Image.fromarray(detected_map)
    # #         for i in range(images.shape[0]):
    #             img_id = mini_batch * args.batch_size + i
                    
    #             images[i].save(os.path.join("/fsx_laion/alvin/pose", f"{img_id}.png"))
    #             detected_map.save(os.path.join("/fsx_laion/alvin/pose", f"{img_id}_pose.png"))
    
            #     print(results)
        # exit(0)
    #     # url_save = url[0].decode('utf-8')
    #     # url_save = url[0]
    #     # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", url_save.split('/')[-1]))
    #     if args.local_rank == 0:
    #         os.makedirs("/fsx_laion/alvin/all", exist_ok=True)
    #         for i in range(images.shape[0]):
    #             img_id = mini_batch * args.batch_size + i
    #             if True:
    #                 # url_save = url[i].decode('utf-8')
    #                 # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", url_save.split('/')[-1]))
    #                 # byte_stream = io.BytesIO()
    #                 # original[i].save(byte_stream, format="JPEG", quality=100)
    #                 # print(type(original[i]), type(text[i]), type(key[i]), type(url[i]), type(json[i]))
    #                 sink.write({
    #                     "jpg": original[i], 
    #                     # "text": text[i].decode('utf-8'),
    #                     # "__key__": key[i].decode('utf-8'), 
    #                     "txt": text[i],
    #                     "__key__": key[i], 
    #                     "__url__": url[i], 
    #                     "json": json[i],
    #                 })
                    
    #                 # original[i].save(os.path.join("/fsx_laion/alvin/all", f"{mini_batch * args.batch_size + i}.png"))
        
    # sink.close()   
    
    
    # # double check that the saving img is correct
    # pipe = StableDiffusionPipeline.from_pretrained("/fsx_laion/alvin/pretrain/stable-diffusion-v1-5")
    # tokenizer = pipe.tokenizer
    # data_args = Args()
    # data_args.train_data = ['/fsx_laion/alvin/Dataset/getty_images_webdataset_human/00016fdf-75da-48f1-8849-7b7500ddc4f9.tar']
    # data_args.train_num_samples = 100
    # data_args.train_data_upsampling_factors = None
    # data_args.workers = args.workers
    # data_args.seed = -1
    # data_args.batch_size = args.batch_size
    # data_args.world_size = 1
    # wds_dataset = get_wds_dataset(data_args,
    #                               preprocess_img=train_transforms,
    #                               is_train=True,
    #                               epoch=0,
    #                               floor=False,
    #                               tokenizer=tokenizer)
    # train_dataloader = wds_dataset.dataloader
    
    # for mini_batch, batch in enumerate(train_dataloader):
    #     images, text_input_ids = batch
    #     if mini_batch == 0 and args.local_rank == 0:
    #         print(images.shape, images.dtype)
    #         print(text_input_ids.shape, text_input_ids.dtype)
    #     os.makedirs("/fsx_laion/alvin/recon", exist_ok=True)
    #     for i in range(images.shape[0]):
    #         pil_img = transforms.ToPILImage()((images[i] + 1) / 2.)
    #         pil_img.save(os.path.join("/fsx_laion/alvin/recon", f"{mini_batch * args.batch_size + i}.png"))
            
        # if args.local_rank == 0:
        #     print(images.shape, images.dtype)
        #     # print(text_input_ids.shape, text_input_ids.dtype)
        #     text = [x.decode('utf-8') for x in text]
        #     key = [x.decode('utf-8') for x in key]
        #     url = [x.decode('utf-8') for x in url]
        #     # json = [x.decode('utf-8') for x in json]
        #     print(text)
        #     print(key)
        #     print(url)
        #     print(json)
        #     exit(0)

if __name__ == '__main__':
    main()
