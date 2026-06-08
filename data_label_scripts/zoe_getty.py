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
    
    # # repo = "isl-org/ZoeDepth"
    # # Zoe_N
    # model_zoe_n = torch.hub.load("/fsx_laion/alvin/ZoeDepth", "ZoeD_N", source="local", weights="/fsx_laion/alvin/pretrain/zoe/ZoeD_M12_N.pt")

    # # Zoe_K
    # # model_zoe_k = torch.hub.load("/fsx_laion/alvin/ZoeDepth", "ZoeD_K", source="local", weights="/fsx_laion/alvin/pretrain/zoe/ZoeD_M12_K.pt")

    # # Zoe_NK
    # # model_zoe_nk = torch.hub.load("/fsx_laion/alvin/ZoeDepth", "ZoeD_NK", source="local", weights="/fsx_laion/alvin/pretrain/zoe/ZoeD_M12_NK.pt")
    
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # zoe = model_zoe_n.to(DEVICE)
    
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
    
    # sink_writers = {}
    # for name in data_args.train_data:
        # sink_writers[name.split('/')[-1]] = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", name.split('/')[-1]))
    # os.makedirs("/fsx_laion/alvin/yolo_filter", exist_ok=True)
    # os.makedirs('/fsx_laion/alvin/visualization/depth-normal/', exist_ok=True)
    os.makedirs('/fsx_laion/alvin/Dataset/getty_human_depnorm', exist_ok=True)
    cnt = 0
    fail_list = []
    for tar_file in tqdm(all_tar_list):
        try:
            writer = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_human_depnorm", tar_file.split('/')[-1]))
            dataset = wds.WebDataset(tar_file)
            sample_count = 0
            for i, sample in enumerate(dataset):
                modified_sample = dict(sample)  # Create a copy of the sample
                
                with io.BytesIO(sample["jpg"]) as stream:
                    try:
                        img = PIL.Image.open(stream)
                    # print(img.info)
                    # compression = img.info.get("compression")
                    # mode = img.mode
                    # quality = img.info.get("quality")
                    # print(compression, mode, quality)
                    # exit(0)
                        img.load()
                        img = img.convert("RGB")
                    except:
                        print("A broken image is encountered, skip")
                        continue

                total_num += 1
                original_height = img.height
                original_width = img.width 
                with torch.no_grad():
                    # import sys
                    # sys.path.insert(0, '../ZoeDepth')
                    # from zoedepth.utils.misc import colorize
                    # # depth_zoe_pil = zoe.infer_pil(img, output_type="pil")   # as 16-bit PIL Image  
                    # colored = colorize(zoe.infer_pil(img))
                    # Image.fromarray(colored).save("/fsx_laion/alvin/visualization/depth.png")
                    # # plt.imsave(f"/fsx_laion/alvin/visualization/depth.png", np.array(depth_zoe_pil), cmap='viridis') 
                    # # depth_zoe_pil.save("/fsx_laion/alvin/visualization/depth.png")
                    # exit(0)
                    # image_file = io.BytesIO()
                    # # Convert the image to the desired format and save it to the file-like object
                    # depth_zoe_pil = depth_zoe_pil.convert('RGB')  # Convert to RGB mode if necessary
                    # depth_zoe_pil.save(image_file, format='PNG')  # Replace 'JPEG' with the desired format

                    # # Get the byte string representation of the image
                    # depth_zoe_pil = image_file.getvalue()
                    # modified_sample["depth"] = depth_zoe_pil
                    
                    # modified_sample["depth"] = depth_zoe_pil.tobytes()
                    
                    normal_img_tensor = normal_trans_totensor(img)[:3].unsqueeze(0).to(device)
                    if normal_img_tensor.shape[1] == 1:
                        normal_img_tensor = normal_img_tensor.repeat_interleave(3,1)

                    normal_output = normal_model(normal_img_tensor).clamp(min=0, max=1)
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
                    # normal_byte = normal_pil.tobytes()
                    # modified_sample["omni_normal"] = normal_byte
                    # norm_recon = Image.frombytes("RGB", (original_height, original_width), normal_byte)
                    # print(normal_pil.tobytes() == norm_recon.tobytes())
                    # exit(0)
                    # normal_pil.save(f"/fsx_laion/alvin/visualization/depth-normal/{i}_normal.png")
                    
                    depth_img_tensor = depth_trans_totensor(img)[:3].unsqueeze(0).to(device)
                    if depth_img_tensor.shape[1] == 1:
                        depth_img_tensor = depth_img_tensor.repeat_interleave(3,1)

                    depth_output = depth_model(depth_img_tensor).clamp(min=0, max=1)
                    
                    depth_output = F.interpolate(depth_output.unsqueeze(0), (original_height, original_width), mode='bicubic').squeeze(0)
                    depth_output = depth_output.clamp(0, 1)
                    depth_output = 1 - depth_output

                    image_file = io.BytesIO()
                    plt.imsave(image_file, depth_output.detach().cpu().squeeze(), cmap='viridis')
                    # depth_pil = transforms.ToPILImage()(depth_output)
                    # image_file = io.BytesIO()
                    # Convert the image to the desired format and save it to the file-like object
                    # depth_pil = depth_pil.convert('RGB')  # Convert to RGB mode if necessary
                    # depth_pil.save(image_file, format='PNG')  # Replace 'JPEG' with the desired format
                    # depth_pil.save(image_file, format='JPEG')  # Replace 'JPEG' with the desired format

                    # Get the byte string representation of the image
                    depth_pil = image_file.getvalue()
                    modified_sample["omni_depth"] = depth_pil
                    # depth_byte = depth_pil.tobytes()
                    # modified_sample["omni_depth"] = depth_byte
                    # depth_output = standardize_depth_map(depth_output)
                    # plt.imsave(f"/fsx_laion/alvin/visualization/depth-normal/{i}_depth.png", depth_output.detach().cpu().squeeze(), cmap='viridis')
                    
                    # import sys
                    # sys.path.insert(0, '../ZoeDepth')
                    # Save raw
                    # from zoedepth.utils.misc import save_raw_16bit
                    # fpath = "/path/to/output.png"
                    # save_raw_16bit(depth, fpath)

                    # Colorize output
                    # from zoedepth.utils.misc import colorize

                    # colored = colorize(depth_numpy)
                    # print(depth_numpy)
                    # print(depth_numpy.shape)
                    # print(colored.shape)
                    # print(depth_numpy.dtype)
                    # exit(0)
                # Add a new attribute to the modified sample
                # modified_sample["new_attribute"] = "new_value"

                # Write the modified sample back to the tar file
                writer.write(modified_sample)
                sample_count += 1

            print("Number of samples in dataset:", sample_count)
            writer.close()
        except:
            os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/getty_human_depnorm', tar_file.split('/')[-1])}")
            fail_list.append(tar_file.split('/')[-1])

    end = time.time()
    
    print(f"Process {len(all_tar_list)} .tar files, totally contain {total_num} images.")  
    max_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)  # Convert to megabytes
    print(f"Peak GPU memory usage: {max_memory:.2f} MB")
    # print(f"The batch size is {args.batch_size}")
    print(f"The overall time cost is {end - start} seconds, avg each tar file process time is {(end - start) / len(all_tar_list)} seconds")
    print(fail_list)
                
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
