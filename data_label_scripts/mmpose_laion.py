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
    # all_tar_list = all_tar_list[args.start : args.end]
    # numbers = [8955, 8956, 8957, 8958, 8959, 8960, 8961, 8962, 8963, 8964, 8965, 8966, 8967, 8968, 8969, 8970, 8971, 8972, 8973, 8974, 8975, 8976, 8977, 8978, 8979, 8980, 8981, 8982, 8983, 8984, 8985, 8986, 8987, 8988, 8989, 8990, 8991, 8992, 8993, 8994, 8995, 8996, 8997, 8998, 8999, 26251, 26252, 26253, 26254, 26255, 26256, 26257, 26258, 26259, 26260, 26261, 26262, 26263, 26264, 26265, 26266, 26267, 26268, 26269, 26270, 26271, 26272, 26273, 26274, 26275, 26276, 26277, 26278, 26279, 26280, 26281, 26282, 26283, 26284, 26285, 26286, 26287, 26288, 26289, 26290, 26291, 26292, 26293, 26294, 26295, 26296, 26297, 26298, 26299, 32113, 32114, 32115, 32116, 32117, 32118, 32119, 32120, 32121, 32122, 32123, 32124, 32125, 32126, 32127, 32128, 32129, 32130, 32131, 32132, 32133, 32134, 32135, 32136, 32137, 32138, 32139, 32140, 32141, 32142, 32143, 32144, 32145, 32146, 32147, 32148, 32149]
    # all_tar_list = [all_tar_list[x] for x in numbers[args.start : args.end]]
    # numbers = []
    # with open("/fsx_laion/alvin/sd4human/new-mmpose-laion-miss.txt", "r") as file:
    #     for line in file:
    #         numbers.append(int(line.strip()))
    numbers = [55155, 63251, 55275, 62451, 63651, 62851]
    all_tar_list = [all_tar_list[x] for x in numbers[args.start : args.end]]

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
    
    # sink_writers = {}
    # for name in data_args.train_data:
        # sink_writers[name.split('/')[-1]] = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", name.split('/')[-1]))
    os.makedirs("/fsx_laion/alvin/Dataset/laion_human_newpose", exist_ok=True)
    # os.makedirs('/fsx_laion/alvin/visualization/mmpose/', exist_ok=True)
    cnt = 0
    fail_list = []
    # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", data_args.train_data[0].split('/')[-1]))
    for tar_file in tqdm(all_tar_list):
        try:
            tar_name = tar_file.split('/')[-1]
            os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/laion_human_newpose', tar_name)}")
            writer = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/laion_human_newpose", tar_name))
            dataset = wds.WebDataset(tar_file)
            dataset_inpaint = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/laion_human_inpaint", tar_name))
            dataset_ldmk = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/laion_human_ldmk", tar_name))
            sample_count = 0
            for sample1, sample2, sample3 in zip(dataset, dataset_inpaint, dataset_ldmk):
                # modified_sample = dict(sample)  # Create a copy of the sample
                modified_sample = {}
                modified_sample["__key__"] = sample1["__key__"]
                modified_sample["__url__"] = sample1["__url__"]
                modified_sample["inpaint"] = sample2["inpaint"]
                modified_sample["location"] = sample2["location"]
                modified_sample["ldmk"] = sample3["ldmk"]
                
                with io.BytesIO(sample1["jpg"]) as stream:
                    try:
                        img = PIL.Image.open(stream)
                        img.load()
                        img = img.convert("RGB")
                    except:
                        print("A broken image is encountered, skip")
                        continue
                    
                with io.BytesIO(sample2["inpaint"]) as stream:
                    try:
                        inpaint = PIL.Image.open(stream)
                        inpaint.load()
                        inpaint = inpaint.convert("RGB")
                    except:
                        print("A broken image is encountered, skip")
                        continue
                    
                # total_num += 1    
                with torch.no_grad():
                    img_list = [np.array(img), np.array(inpaint)]
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
                    
                    result = next(result_generator)
                    i_wholebody_bbox = result['predictions'][0].pred_instances.bboxes
                    i_wholebody_bbox_score = result['predictions'][0].pred_instances.bbox_scores
                    i_wholebody_kp = result['predictions'][0].pred_instances.keypoints
                    i_wholebody_kp_score = result['predictions'][0].pred_instances.keypoint_scores
                    modified_sample["new_i_wholebody_bbox"] = i_wholebody_bbox.astype(np.float32).tobytes()
                    modified_sample["new_i_wholebody_bbox_score"] = i_wholebody_bbox_score.astype(np.float32).tobytes()
                    modified_sample["new_i_wholebody_kp"] = i_wholebody_kp.astype(np.float32).tobytes()
                    modified_sample["new_i_wholebody_kp_score"] = i_wholebody_kp_score.astype(np.float32).tobytes()
                    # print(len(body_kp))
                    # if body_kp.shape[0] != 1:
                    #     # img.save(f"/fsx_laion/alvin/visualization/mmpose/{i}.png")
                    #     print(body_kp.shape)
                    # print(result['predictions'][0].pred_instances.keypoint_scores[0].dtype)
                    # exit(0)
                    # serialized_data = body_kp.tobytes()
                    # restored_data = np.frombuffer(serialized_data).reshape(17, 2)
                    # print(body_kp == restored_data)
                    # print(body_kp[0])
                    # print(restored_data[0])
                    # exit(0)
                    # modified_sample["body_kp"] = result['predictions'][0].pred_instances.keypoints[0].astype(np.float32).tobytes()
                    # modified_sample["body_kpconf"] = result['predictions'][0].pred_instances.keypoint_scores[0].astype(np.float32).tobytes()
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
                    
                    result = next(result_generator)
                    i_body_bbox = result['predictions'][0].pred_instances.bboxes
                    i_body_bbox_score = result['predictions'][0].pred_instances.bbox_scores
                    i_body_kp = result['predictions'][0].pred_instances.keypoints
                    i_body_kp_score = result['predictions'][0].pred_instances.keypoint_scores
                    modified_sample["new_i_body_bbox"] = i_body_bbox.astype(np.float32).tobytes()
                    modified_sample["new_i_body_bbox_score"] = i_body_bbox_score.astype(np.float32).tobytes()
                    modified_sample["new_i_body_kp"] = i_body_kp.astype(np.float32).tobytes()
                    modified_sample["new_i_body_kp_score"] = i_body_kp_score.astype(np.float32).tobytes()
                    
                    # face_generator = face_inferencer(np.array(img), return_datasample=True)
                    # face_result = next(face_generator)
                    # modified_sample["face_kp"] = face_result['predictions'][0].pred_instances.keypoints.astype(np.float32).tobytes() # (n, 98, 2)
                    # modified_sample["face_kpconf"] = face_result['predictions'][0].pred_instances.keypoint_scores.astype(np.float32).tobytes() # (n, 98)
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
                    
                    result = next(result_generator)
                    i_face_bbox = result['predictions'][0].pred_instances.bboxes
                    i_face_bbox_score = result['predictions'][0].pred_instances.bbox_scores
                    i_face_kp = result['predictions'][0].pred_instances.keypoints
                    i_face_kp_score = result['predictions'][0].pred_instances.keypoint_scores
                    modified_sample["new_i_face_bbox"] = i_face_bbox.astype(np.float32).tobytes()
                    modified_sample["new_i_face_bbox_score"] = i_face_bbox_score.astype(np.float32).tobytes()
                    modified_sample["new_i_face_kp"] = i_face_kp.astype(np.float32).tobytes()
                    modified_sample["new_i_face_kp_score"] = i_face_kp_score.astype(np.float32).tobytes()
                    
                    # hand_generator = hand_inferencer(np.array(img), return_datasample=True)
                    # hand_result = next(hand_generator)
                    # modified_sample["hand_kp"] = hand_result['predictions'][0].pred_instances.keypoints.astype(np.float32).tobytes() # (n, 21, 2)
                    # modified_sample["hand_kpconf"] = hand_result['predictions'][0].pred_instances.keypoint_scores.astype(np.float32).tobytes() # (n, 21)
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
                    
                    result = next(result_generator)
                    i_hand_bbox = result['predictions'][0].pred_instances.bboxes
                    i_hand_bbox_score = result['predictions'][0].pred_instances.bbox_scores
                    i_hand_kp = result['predictions'][0].pred_instances.keypoints
                    i_hand_kp_score = result['predictions'][0].pred_instances.keypoint_scores
                    modified_sample["new_i_hand_bbox"] = i_hand_bbox.astype(np.float32).tobytes()
                    modified_sample["new_i_hand_bbox_score"] = i_hand_bbox_score.astype(np.float32).tobytes()
                    modified_sample["new_i_hand_kp"] = i_hand_kp.astype(np.float32).tobytes()
                    modified_sample["new_i_hand_kp_score"] = i_hand_kp_score.astype(np.float32).tobytes()
                    
                # Add a new attribute to the modified sample
                # modified_sample["new_attribute"] = "new_value"
                
                # Write the modified sample back to the tar file
                writer.write(modified_sample)
                # sample_count += 1

            # print("Number of samples in dataset:", sample_count)
            writer.close()
        except:
            os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/laion_human_newpose', tar_file.split('/')[-1])}")
            fail_list.append(tar_file.split('/')[-1])
        
        # exit(0)
        # try:
        # data_args.train_data = [tar_file]
        # wds_dataset = get_wds_dataset_filter(data_args, identity)
        # train_dataloader = wds_dataset.dataloader
        # # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/laion2B-en-aesthetic-4.5plus-512_human_structural", tar_file.split('/')[-1]))
        # with torch.no_grad():
        #     for mini_batch, batch in enumerate(train_dataloader):
        #         # try:
        #         original, images, text, key, url, json = batch
        #         for i, each in enumerate(images):            
        #             result_generator = body_inferencer(np.array(each), return_vis=True, return_datasample=True, radius=12, thickness=4)
        #             # print(result_generator)
        #             result = next(result_generator)
                    # print(result['predictions'][0])
                    # print(result['predictions'][0].pred_instances)
                    # print(result['predictions'][0].pred_instances.keypoints.shape) # (1, 17, 2)
                    # print(result['predictions'][0].pred_instances.keypoint_scores.shape) # (1, 17)
                    # try:
                    # assert len(result['predictions']) == 1
                    # assert result['predictions'][0].pred_instances.keypoints.shape == (1, 17, 2)
                    # assert result['predictions'][0].pred_instances.keypoint_scores.shape == (1, 17)
                    # except:
                        # print(result['predictions'][0].pred_instances.keypoints.shape) # (1, 17, 2)
                        # print(result['predictions'][0].pred_instances.keypoint_scores.shape) # (1, 17)
                    
                    # exit(0)
                    # print(type(result['visualization'][0]))
                    # img = Image.fromarray(mmcv.rgb2bgr(result['visualization'][0]))
                    # img.convert('RGB')
                    # img.save(f'/fsx_laion/alvin/visualization/mmpose/{i + mini_batch * args.batch_size}.png')
                    
                    # face_generator = face_inferencer(np.array(each), return_vis=True, return_datasample=True, radius=12, thickness=4)
                    # face_result = next(face_generator)
                    # print(face_result['predictions'][0])
                    # print(face_result['predictions'][0].pred_instances.keypoints.shape) # (n, 98, 2)
                    # print(face_result['predictions'][0].pred_instances.keypoint_scores.shape) # (n, 98)
                    # try:
                    #     assert face_result['predictions'][0].pred_instances.keypoints.shape == (1, 98, 2)
                    #     assert face_result['predictions'][0].pred_instances.keypoint_scores.shape == (1, 98)
                    # except:
                    #     print(face_result['predictions'][0].pred_instances.keypoints.shape) # (1, 98, 2)
                    #     print(face_result['predictions'][0].pred_instances.keypoint_scores.shape) # (1, 98)

                    # face_img = Image.fromarray(mmcv.rgb2bgr(face_result['visualization'][0]))
                    # face_img.save(f'/fsx_laion/alvin/visualization/mmpose/{i + mini_batch * args.batch_size}_face.png')
                    
                    # hand_generator = hand_inferencer(np.array(each), return_vis=True, return_datasample=True, radius=12, thickness=4)
                    # hand_result = next(hand_generator)
                    # print(hand_result['predictions'][0].pred_instances.keypoints.shape) # (n, 21, 2)
                    # print(hand_result['predictions'][0].pred_instances.keypoint_scores.shape) # (n, 21)
                    # try:
                    #     assert hand_result['predictions'][0].pred_instances.keypoints.shape == (1, 21, 2)
                    #     assert hand_result['predictions'][0].pred_instances.keypoint_scores.shape == (1, 21)
                    # except:
                    #     print(hand_result['predictions'][0].pred_instances.keypoints.shape) # (1, 21, 2)
                    #     print(hand_result['predictions'][0].pred_instances.keypoint_scores.shape) # (1, 21)
                    # hand_img = Image.fromarray(mmcv.rgb2bgr(hand_result['visualization'][0]))
                    # hand_img.save(f'/fsx_laion/alvin/visualization/mmpose/{i + mini_batch * args.batch_size}_hand.png')
                    
                    # cnt += 1
                    # if cnt == 100:
                    #     exit(0)

                        # sink = sink_writers[url[0].split('/')[-1]]
                        # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", url[0].split('/')[-1]))
                        # if len(images) != args.batch_size:
                        #     print(len(images))
                        #     print(url)
                        # assert url[0] == url[-1]
                        # continue
                        # inputs = image_processor(images=images, return_tensors="pt").to("cuda")
                        # outputs = model(**inputs)

                        # # model predicts bounding boxes and corresponding COCO classes
                        # logits = outputs.logits
                        # bboxes = outputs.pred_boxes

                        # # print results
                        # target_sizes = torch.stack([torch.tensor(images[i].size[::-1]) for i in range(len(images))])
                        # results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)
                        # for i, result in enumerate(results):
                        #     total_num += 1
                        #     person_num = 0
                        #     area_ratio = 0.
                        #     scores = result["scores"]
                        #     labels = result["labels"]
                        #     boxs = result["boxes"]
                        #     for score, label, box in zip(scores, labels, boxs):
                        #         if model.config.id2label[label.item()] == "person":
                        #             box = [round(i, 2) for i in box.tolist()]
                        #             area_ratio = max(area_ratio, (box[2] - box[0]) * (box[3] - box[1]) / (target_sizes[i][0] * target_sizes[i][1]))
                        #             # print(area_ratio.item())
                        #             person_num += 1
                                
                        #     if person_num > 0 and person_num < 4 and area_ratio > 0.1:
                        #         success_num += 1
                        #         sink.write({
                        #             "jpg": original[i], 
                        #             # "text": text[i].decode('utf-8'),
                        #             # "__key__": key[i].decode('utf-8'), 
                        #             "txt": text[i],
                        #             "__key__": key[i], 
                        #             "__url__": url[i], 
                        #             "json": json[i],
                        #         })
                                # img_id = mini_batch * args.batch_size + i                   
                                # images[i].save(os.path.join("/fsx_laion/alvin/yolo_filter", f"{img_id}.png"))
                                    # break
                            # if person_num > 0: print(person_num)
                            #     print(1)
                    # except:
                        # pass
                # sink.close()
        # except:
            # pass
            # os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/laion2B-en-aesthetic-4.5plus-512_human_structural', tar_file.split('/')[-1])}")
            # print(f"{tar_file.split('/')[-1]} fails")
            
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
