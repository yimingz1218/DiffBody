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
# from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
import time

from openclip.training.data import get_wds_dataset, get_wds_dataset_filter, tarfile_to_samples_nothrow, get_wds_dataset_img


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
    # # data_args.train_data = []
    # # all_tar_list = [os.path.join(data_folder, x)
    # #                 for x in os.listdir(data_folder) if
    # #                 x.endswith('.tar')]
    # # data_args.train_data = ['/fsx_laion/getty_images_webdataset/00016fdf-75da-48f1-8849-7b7500ddc4f9.tar']
    for data_folder in args.data_path:
        all_tar_list += [os.path.join(data_folder, x)
                                 for x in sorted(os.listdir(data_folder)) if
                                 x.endswith('.tar')]  # open('s3_urls.txt').read().splitlines()
    # all_tar_list = all_tar_list[args.start : args.end]
    # # numbers = [333, 334, 335, 355, 383, 385, 390, 397, 435, 436, 444, 463, 590, 595, 1243, 1244, 1259, 12929, 13029, 13179, 13279, 13379, 13479, 13529, 13729, 13979, 14029, 14329]
    # # all_tar_list = [all_tar_list[x] for x in numbers[args.start : args.end]]
    # # all_tar_list = [all_tar_list[x] for x in numbers]
    # numbers = []
    # with open("/fsx_laion/alvin/sd4human/blipft-getty-miss.txt", "r") as file:
    #     for line in file:
    #         numbers.append(int(line.strip()))
    # numbers = [55155, 63251, 55275, 62451, 63651, 62851]
    numbers = [4050, 4051, 4052, 4053, 4054, 4090, 4091, 4092, 4093, 4094, 6849, 7862, 8850, 8851, 8852, 20753, 20754, 20755, 20756, 20757, 20773, 20774, 20775, 20776, 20777]
    all_tar_list = [all_tar_list[x] for x in numbers[args.start : args.end]]

    print(f'Found {len(all_tar_list)} .tar files in {args.data_path}')
    
    # os.makedirs("/fsx_laion/alvin/Dataset/laion2B-en-aesthetic-4.5plus-512_human_structural", exist_ok=True)
    # total_num = success_num = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    
    from lavis.models import load_model_and_preprocess
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
    )
    # exit(0)
    ckpt_path = "/fsx_laion/alvin/pretrain/blip2/human_latest.pth"

    ckpt = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt["model"], strict=False)
    model.max_txt_len = 300
    model.eval()
    
    # identity = lambda x: x
    
    # class Args:
    #     pass  # to use open clip api
    # data_args = Args()
    # # data_args.val_data = []

    # # for data_folder in args.data_path:
    # #     data_list = [os.path.join(data_folder, x)
    # #                             for x in sorted(os.listdir(data_folder)) if
    # #                             x.endswith('.tar')]  # open('s3_urls.txt').read().splitlines()
    
    # #     data_args.val_data += data_list
    # # data_args.val_data = data_args.val_data[args.start : args.end]
    # # print(f'Found {len(data_args.val_data)} .tar files in {args.test_data_dir}')
    # # data_args.train_num_samples = 400000000
    # data_args.train_data_upsampling_factors = None
    # data_args.val_num_samples = None
    # data_args.batch_size = 1
    # # data_args.world_size = torch.distributed.get_world_size()
    # # print(torch.distributed.get_world_size())
    # data_args.workers = 12
    # data_args.seed = -1
    # test_dataset = get_wds_dataset_img(data_args,
    #                               preprocess_img=vis_processors["eval"],
    #                               is_train=False,
    #                               epoch=0,
    #                               floor=False,
    #                               )
    # train_dataset = train_dataset.with_length(300000000)
    # test_dataloader = test_dataset.dataloader
    
    # from transformers import Blip2Processor, Blip2ForConditionalGeneration

    # processor = Blip2Processor.from_pretrained("/fsx_laion/alvin/pretrain/blip2-opt-6.7b-coco")
    # model = Blip2ForConditionalGeneration.from_pretrained(
    #     "/fsx_laion/alvin/pretrain/blip2-opt-6.7b-coco", torch_dtype=torch.float16
    # )
    # model.to(device)

    # sink_writers = {}
    # for name in data_args.train_data:
        # sink_writers[name.split('/')[-1]] = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", name.split('/')[-1]))
    os.makedirs("/fsx_laion/alvin/Dataset/getty_human_blipft", exist_ok=True)
    # os.makedirs('/fsx_laion/alvin/visualization/mmpose/', exist_ok=True)
    cnt = 0
    fail_list = []
    # sink = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_images_webdataset_human", data_args.train_data[0].split('/')[-1]))
    for tar_file in tqdm(all_tar_list):
        try:
            tar_name = tar_file.split('/')[-1]
            os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/getty_human_blipft', tar_name)}")
            writer = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_human_blipft", tar_name))
            dataset = wds.WebDataset(tar_file)
            for sample in dataset:
                modified_sample = dict(sample)
                with io.BytesIO(sample["jpg"]) as stream:
                    try:
                        img = PIL.Image.open(stream)
                        img.load()
                        img = img.convert("RGB")
                    except:
                        print("A broken image is encountered, skip")
                        continue
                    
                image = vis_processors["eval"](img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    result = model.generate({"image": image}, 
                            num_beams=5,
                            max_length=300,
                            min_length=10,
                            repetition_penalty=3.0,
                            num_captions=1)
                
                modified_sample["blip_humanft"] = result[0]
                writer.write(modified_sample)
            writer.close()
        # try:
        # tar_name = tar_file.split('/')[-1]
        # os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/getty_human_blipft', tar_name)}")
        # data_args.val_data = [tar_file]
        # test_dataset = get_wds_dataset_img(data_args,
        #                           preprocess_img=vis_processors["eval"],
        #                         #   preprocess_img=identity,
        #                           is_train=False,
        #                           epoch=0,
        #                           floor=False,
        #                           )
        # test_dataloader = test_dataset.dataloader
        # # cnt = 0
        # blip_results = []
        # for i, batch in enumerate(test_dataloader):
        #     # image = torch.stack(batch, dim=0)
        #     image = batch[0].to(device)
        #     # print(image.shape)
        # #     cnt += len(image[0])
        # # print(cnt)
        # # cnt = 0
        # # dataset = wds.WebDataset(tar_file)
        # # for sample in dataset:
        # #     cnt += 1
        # # print(cnt)
        #     with torch.no_grad():
        #         result = model.generate({"image": image}, 
        #                 num_beams=5,
        #                 max_length=300,
        #                 min_length=10,
        #                 repetition_penalty=3.0,
        #                 num_captions=1)
        #         for caption in result:
        #             blip_results.append(caption)
        # print(blip_results)
        # print(len(blip_results))
        # # writer = wds.TarWriter(os.path.join("/fsx_laion/alvin/Dataset/getty_human_blipft", tar_name))
        # # dataset = wds.WebDataset(tar_file)
        # # dataset_inpaint = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/getty_human_inpaint", tar_name))
        # # dataset_ldmk = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/getty_human_ldmk", tar_name))
        # # sample_count = 0
        # for sample in dataset:
        #     modified_sample = dict(sample)  # Create a copy of the sample
        #     # modified_sample = {}
        #     # modified_sample["__key__"] = sample["__key__"]
        #     # modified_sample["__url__"] = sample["__url__"]
        #     # modified_sample["inpaint"] = sample2["inpaint"]
        #     # modified_sample["location"] = sample2["location"]
        #     # modified_sample["ldmk"] = sample1["ldmk"]
        #     # print(sample["__key__"])
            
        #     with io.BytesIO(sample["jpg"]) as stream:
        #         try:
        #             img = PIL.Image.open(stream)
        #             img.load()
        #             img = img.convert("RGB")
        #         except:
        #             print("A broken image is encountered, skip")
        #             continue
                
        #     # inputs = processor(images=img, return_tensors="pt").to(device, torch.float16)

        #     # generated_ids = model.generate(
        #     #     **inputs,
        #     #     num_beams=5,
        #     #     max_length=300,
        #     #     min_length=10,
        #     #     repetition_penalty=3.0,
        #     #     num_captions=1
        #     # )
        #     # text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        #     # print(text)
        #     image = vis_processors["eval"](img).unsqueeze(0).to(device)
            
        #     with torch.no_grad():
        #         result = model.generate({"image": image}, 
        #                 num_beams=5,
        #                 max_length=300,
        #                 min_length=10,
        #                 repetition_penalty=3.0,
        #                 num_captions=1)
        #         print(result)
                            
            # Write the modified sample back to the tar file
    #         writer.write(modified_sample)
    #     writer.close()
        except:
            os.system(f"rm -f {os.path.join('/fsx_laion/alvin/Dataset/getty_human_blipft', tar_file.split('/')[-1])}")
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
