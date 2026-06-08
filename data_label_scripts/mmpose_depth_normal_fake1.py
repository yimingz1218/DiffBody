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

from transformers import CLIPTextModel, CLIPTokenizer, DPTFeatureExtractor, DPTForDepthEstimation

from contextlib import redirect_stdout
from contextlib import contextmanager
import sys
import torch.nn.functional as F
import matplotlib.pyplot as plt

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

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
    # for data_folder in args.data_path:
    #     all_tar_list += [os.path.join(data_folder, x)
    #                              for x in sorted(os.listdir(data_folder)) if
    #                              x.endswith('.tar')]  # open('s3_urls.txt').read().splitlines()
        
    # numbers = []
    # with open("/fsx_laion/alvin/sd4human/mmpose-depth-normal-fake1-miss.txt", "r") as file:
    #     for line in file:
    #         numbers.append(int(line.strip()))
    # # numbers2 = [num + 25600 for num in numbers]
    # all_tar_list = [all_tar_list[x] for x in numbers[args.start : args.end]]
    # all_tar_list = all_tar_list[args.start : args.end]
    # numbers = [333, 334, 335, 355, 383, 385, 390, 397, 435, 436, 444, 463, 590, 595, 1243, 1244, 1259, 12929, 13029, 13179, 13279, 13379, 13479, 13529, 13729, 13979, 14029, 14329]
    # all_tar_list = [all_tar_list[x] for x in numbers[args.start : args.end]]
    # all_tar_list = [all_tar_list[x] for x in numbers]
    all_tar_list = ['/nfs/del22/fake_data_sdxl_general_090523/5ce357929-3263-40b6-afdb-a12875f5d3a4.tar']

    print(f'Found {len(all_tar_list)} .tar files in {args.data_path}')
    
    # os.makedirs("/fsx_laion/alvin/Dataset/laion2B-en-aesthetic-4.5plus-512_human_structural", exist_ok=True)
    total_num = success_num = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    
    from mmpose.apis import MMPoseInferencer
    import mmcv
    # body_inferencer = MMPoseInferencer("human")
    whole_inferencer = MMPoseInferencer(pose2d='/fsx_laion/alvin/mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py',
                                       pose2d_weights='/fsx_laion/alvin/pretrain/ViTPose/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth',
                                    #    det_model='rtmdet-m',
                                    #     det_weights="/fsx_laion/alvin/pretrain/ViTPose/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
                                    )
    body_inferencer = MMPoseInferencer(pose2d='/fsx_laion/alvin/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192.py',
                                       pose2d_weights='/fsx_laion/alvin/pretrain/ViTPose/td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192-ffd48c05_20230314.pth',
                                       scope="mmpose"
                                    #    det_model='/fsx_laion/alvin/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
                                    #    det_weights="/fsx_laion/alvin/pretrain/ViTPose/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
                                    )
    # body_inferencer = MMPoseInferencer(pose2d='/fsx_laion/alvin/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
    #                                    pose2d_weights='/fsx_laion/alvin/pretrain/ViTPose/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
    #                                    det_model='/fsx_laion/alvin/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
    #                                    det_weights="/fsx_laion/alvin/pretrain/ViTPose/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")
    face_inferencer = MMPoseInferencer(pose2d='/fsx_laion/alvin/mmpose/configs/face_2d_keypoint/rtmpose/coco_wholebody_face/rtmpose-m_8xb32-60e_coco-wholebody-face-256x256.py',
                                       pose2d_weights='/fsx_laion/alvin/pretrain/ViTPose/rtmpose-m_simcc-coco-wholebody-face_pt-aic-coco_60e-256x256-62026ef2_20230228.pth',
                                    #    det_model='rtmdet-m',
                                    #     det_weights="/fsx_laion/alvin/pretrain/ViTPose/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
                                    )
    hand_inferencer = MMPoseInferencer(pose2d='/fsx_laion/alvin/mmpose/configs/hand_2d_keypoint/rtmpose/coco_wholebody_hand/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py',
                                       pose2d_weights='/fsx_laion/alvin/pretrain/ViTPose/rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco_210e-256x256-99477206_20230228.pth',
                                    #    det_model='rtmdet-m',
                                    #     det_weights="/fsx_laion/alvin/pretrain/ViTPose/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
                                    )
    
    feature_extractor = DPTFeatureExtractor.from_pretrained("/fsx_laion/alvin/pretrain/stable-diffusion-2-depth", subfolder="feature_extractor")
    depth_estimator = DPTForDepthEstimation.from_pretrained("/fsx_laion/alvin/pretrain/stable-diffusion-2-depth", subfolder="depth_estimator").to(device)
    
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
    os.makedirs("/fsx_laion/alvin/Dataset/fake_sdxl_general", exist_ok=True)
    # os.makedirs('/fsx_laion/alvin/visualization/mmpose/', exist_ok=True)
    cnt = 0
    fail_list = []
    # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", data_args.train_data[0].split('/')[-1]))
    for tar_file in tqdm(all_tar_list):
        try:
            tar_name = tar_file.split('/')[-1]
            os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/fake_sdxl_general', tar_name)}")
            writer = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/fake_sdxl_general", tar_name))
            dataset = wds.WebDataset(tar_file)
            # dataset_inpaint = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/getty_human_inpaint", tar_name))
            # dataset_ldmk = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/getty_human_ldmk", tar_name))
            sample_count = 0
            for i, sample in enumerate(dataset):
                modified_sample = dict(sample)  # Create a copy of the sample
                
                try:
                    with io.BytesIO(sample["jpg"]) as stream:
                        try:
                            img = PIL.Image.open(stream)
                            img.load()
                            img = img.convert("RGB")
                        except:
                            print("A broken image is encountered, skip")
                            continue
                except:
                    continue
                    
                original_height = img.height
                original_width = img.width 
                    
                # total_num += 1    
                with torch.no_grad():
                    with suppress_output():
                        img_list = [np.array(img)]
                        result_generator = whole_inferencer(img_list, return_datasample=True)
                        result = next(result_generator)
                        # print(result['predictions'][0])
                        # exit(0)
                        wholebody_bbox = result['predictions'][0].pred_instances.bboxes
                        wholebody_bbox_score = result['predictions'][0].pred_instances.bbox_scores
                        wholebody_kp = result['predictions'][0].pred_instances.keypoints
                        wholebody_kp_score = result['predictions'][0].pred_instances.keypoint_scores
                        modified_sample["new_wholebody_bbox"] = wholebody_bbox.astype(np.float32).tobytes()
                        modified_sample["new_wholebody_bbox_score"] = wholebody_bbox_score.astype(np.float32).tobytes()
                        modified_sample["new_wholebody_kp"] = wholebody_kp.astype(np.float32).tobytes()
                        modified_sample["new_wholebody_kp_score"] = wholebody_kp_score.astype(np.float32).tobytes()

                        result_generator = body_inferencer(img_list, return_datasample=True)
                        result = next(result_generator)
                        # print(len(result))
                        # print(result['predictions'][0])
                        # exit(0)
                        body_bbox = result['predictions'][0].pred_instances.bboxes
                        body_bbox_score = result['predictions'][0].pred_instances.bbox_scores
                        body_kp = result['predictions'][0].pred_instances.keypoints
                        body_kp_score = result['predictions'][0].pred_instances.keypoint_scores
                        modified_sample["new_body_bbox"] = body_bbox.astype(np.float32).tobytes()
                        modified_sample["new_body_bbox_score"] = body_bbox_score.astype(np.float32).tobytes()
                        modified_sample["new_body_kp"] = body_kp.astype(np.float32).tobytes()
                        modified_sample["new_body_kp_score"] = body_kp_score.astype(np.float32).tobytes()
                    
                        result_generator = face_inferencer(img_list, return_datasample=True)
                        result = next(result_generator)
                        # print(len(result))
                        # print(result['predictions'][0])
                        # exit(0)
                        face_bbox = result['predictions'][0].pred_instances.bboxes
                        face_bbox_score = result['predictions'][0].pred_instances.bbox_scores
                        face_kp = result['predictions'][0].pred_instances.keypoints
                        face_kp_score = result['predictions'][0].pred_instances.keypoint_scores
                        modified_sample["new_face_bbox"] = face_bbox.astype(np.float32).tobytes()
                        modified_sample["new_face_bbox_score"] = face_bbox_score.astype(np.float32).tobytes()
                        modified_sample["new_face_kp"] = face_kp.astype(np.float32).tobytes()
                        modified_sample["new_face_kp_score"] = face_kp_score.astype(np.float32).tobytes()
                    
                        result_generator = hand_inferencer(img_list, return_datasample=True)
                        result = next(result_generator)
                        # print(len(result))
                        # print(result['predictions'][0])
                        # exit(0)
                        hand_bbox = result['predictions'][0].pred_instances.bboxes
                        hand_bbox_score = result['predictions'][0].pred_instances.bbox_scores
                        hand_kp = result['predictions'][0].pred_instances.keypoints
                        hand_kp_score = result['predictions'][0].pred_instances.keypoint_scores
                        modified_sample["new_hand_bbox"] = hand_bbox.astype(np.float32).tobytes()
                        modified_sample["new_hand_bbox_score"] = hand_bbox_score.astype(np.float32).tobytes()
                        modified_sample["new_hand_kp"] = hand_kp.astype(np.float32).tobytes()
                        modified_sample["new_hand_kp_score"] = hand_kp_score.astype(np.float32).tobytes()
                        
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
                    
                    normal_img_tensor = normal_trans_totensor(img)[:3].unsqueeze(0).to(device)
                    if normal_img_tensor.shape[1] == 1:
                        normal_img_tensor = normal_img_tensor.repeat_interleave(3,1)

                    normal_output = normal_model(normal_img_tensor).clamp(min=0, max=1)

                    trans_back = transforms.Compose([
                                    transforms.ToPILImage(),
                                    # transforms.Resize((512, 512), interpolation=PIL.Image.BILINEAR),
                                    ])
                    normal_pil = trans_back(normal_output[0])
                    # normal_pil.save(f"/fsx_laion/alvin/visualization/depth-normal/{i}_normal.png")
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
                    
                    # depth_output = F.interpolate(depth_output.unsqueeze(0), (original_height, original_width), mode='bicubic').squeeze(0)
                    # depth_output = depth_output.clamp(0, 1)
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
                
                # if i >= 20:
                #     break
                # Add a new attribute to the modified sample
                # modified_sample["new_attribute"] = "new_value"
                
                # Write the modified sample back to the tar file
                writer.write(modified_sample)
                # sample_count += 1
                # if sample_count >= 10:
                #     break

            # print("Number of samples in dataset:", sample_count)
            writer.close()
        except:
            os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/fake_sdxl_general', tar_file.split('/')[-1])}")
            fail_list.append(tar_file.split('/')[-1])
            
    # for writer in sink_writers.values():
    #     writer.close()
    end = time.time()
    
    # print(f"Process {len(all_tar_list)} .tar files, totally contain {total_num} images.")  
    # max_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)  # Convert to megabytes
    # print(f"Peak GPU memory usage: {max_memory:.2f} MB")
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
