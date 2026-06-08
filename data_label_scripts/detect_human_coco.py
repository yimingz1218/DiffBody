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

class CustomDataset(Dataset):
    def __init__(self, json_path='/fsx_laion/alvin/Dataset/coco/img_text_train2014.json', image_processor=None):
        self.image_data = []
        self.text_data = []
        self.id_list = []
        self.image_processor = image_processor

        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
        
        for image_id, caption in json_data.items():
            self.image_data.append(f'/fsx_laion/alvin/Dataset/coco/train2014/COCO_train2014_{int(image_id):012d}.jpg')
            self.text_data.append(caption)
            self.id_list.append(int(image_id))
            
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        image_path = self.image_data[idx]
        image = Image.open(image_path).convert("RGB")
        
        resize = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)
        image = resize(image)
        
        if self.image_processor:
            image = self.image_processor(images=image, return_tensors="pt")

        return image['pixel_values'][0], self.text_data[idx], self.id_list[idx]

def parse():
    parser = argparse.ArgumentParser(description='PyTorch DDP Training')
    
    args = parser.parse_args()
    return args

def collate_fn(batch):
    images, texts, ids = zip(*batch)
    images = torch.stack(images)
    
    return images, texts, ids

def main():

    args = parse()

    cudnn.benchmark = True

    from transformers import YolosImageProcessor, YolosForObjectDetection
    
    model = YolosForObjectDetection.from_pretrained('/fsx_laion/alvin/pretrain/yolos-tiny').to('cuda')
    image_processor = YolosImageProcessor.from_pretrained("/fsx_laion/alvin/pretrain/yolos-tiny")
    
    # os.makedirs("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", exist_ok=True)
    total_num = success_num = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = CustomDataset(image_processor=image_processor)

    # Create a DataLoader for batching and shuffling
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, collate_fn=collate_fn)
    
    output = {}

    # Now you can iterate through the dataloader to get batches of data
    for iter, batch in enumerate(tqdm(dataloader)):
        images, text, id = batch

        images = images.to(device)
        outputs = model(images)

        # model predicts bounding boxes and corresponding COCO classes
        logits = outputs.logits
        bboxes = outputs.pred_boxes

        # print results
        # target_sizes = torch.stack([torch.tensor(images[i].size[::-1]) for i in range(len(images))])
        target_sizes = torch.stack([torch.tensor((512, 512)) for i in range(len(images))]).to(device)
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
                                
                    if person_num > 0 and person_num < 4 and area_ratio > 0.2:
                        success_num += 1
                        output[str(id[i])] = text[i]
                        
    with open('/fsx_laion/alvin/Dataset/coco/img_text_train2014-human.json', "w") as json_file:
        json.dump(output, json_file)                    
    print(success_num)
    print(total_num)
    print(1.0 * success_num / total_num)
        #                         sink.write({
        #                             "jpg": original[i], 
        #                             # "text": text[i].decode('utf-8'),
        #                             # "__key__": key[i].decode('utf-8'), 
        #                             "txt": text[i],
        #                             "__key__": key[i], 
        #                             "__url__": url[i], 
        #                             "json": json[i],
        #                         })
        #                         # img_id = mini_batch * args.batch_size + i                   
        #                         # images[i].save(os.path.join("/fsx_laion/alvin/yolo_filter", f"{img_id}.png"))
        #                             # break
        #                     # if person_num > 0: print(person_num)
        #                     #     print(1)
        #             except:
        #                 pass
        #         sink.close()
        # except:
        #     os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/getty_images_webdataset_human', tar_file.split('/')[-1])}")
        #     print(f"{tar_file.split('/')[-1]} fails")

if __name__ == '__main__':
    main()
