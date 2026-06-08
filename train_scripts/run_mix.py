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

from diffusers import StableDiffusionPipeline, StableDiffusionPipelineAda, LMSDiscreteScheduler
from diffusers.models import AutoencoderKL
from unetdeep import UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, \
    DEISMultistepScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, \
    PNDMScheduler

from openclip.training.data import get_wds_dataset

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


def calc_SNR(scheduler):
    def add_noise_DDIM(
            self,
            original_samples: torch.FloatTensor,
            noise: torch.FloatTensor,
            timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        setattr(noisy_samples, 'SNR', (sqrt_alpha_prod / sqrt_one_minus_alpha_prod) ** 2)
        return noisy_samples

    if scheduler.__class__ in [DDIMScheduler, DDPMScheduler, PNDMScheduler]:
        ## https://stackoverflow.com/a/54662690
        scheduler.original_add_noise = scheduler.add_noise
        scheduler.add_noise = partial(add_noise_DDIM, scheduler)
    else:
        raise NotImplementedError
    return scheduler


class CsvDataset(Dataset):
    def __init__(self, input_filename='openprompts.csv'):
        self.df = pd.read_csv(input_filename, keep_default_na=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.df.iloc[idx, 0]
        return prompt


class TextDataset(Dataset):
    def __init__(self, data_path='/fsx/laion/data/coyo_openprompt_stablediffusion_midjourney/mix', bs=12):
        self.data_path = data_path
        self.bs = bs

    def __len__(self):
        return 757761676
        # return 757762

    def __getitem__(self, idx):
        sample = idx % 1000
        file = os.path.join(self.data_path, 'mix-1k-' + str(f"{idx // 1000 :07d}") + '.txt')
        return linecache.getline(file, sample)[:-1]

        # file = os.path.join(self.data_path, 'mix-1k-' + str(f"{idx :07d}") + '.txt')
        # with open(file, 'r') as prompts:
        #     lines = prompts.readlines()
        #     selected = random.sample(lines, self.bs)
        #     prompt = [item.rstrip() for item in selected]
        # return tuple(prompt)


def hook_prune(mask):
    def hook_prune_irregular(module, input):
        module.weight.data *= mask
        # module.weight = module.weight * mask

    return hook_prune_irregular


def inherit_weights(student, teacher):
    teacher_weights = teacher.state_dict()
    student_weights = student.state_dict()
    inherited_weights = {}

    for item in student_weights:
        # there can be teacher layers that are not in student model, but all student layers should have a teacher weight
        if teacher_weights[item].size() != student_weights[item].size():
            new_weight = teacher_weights[item]
            size = student_weights[item].size()
            for i in range(len(size)):
                # artifacts here, to perfectly inherit teacher weight, need to handle [concat] !!!!!!
                new_weight = new_weight.index_select(dim=i, index=torch.tensor(range(size[i])))
            inherited_weights[item] = new_weight
        else:
            inherited_weights[item] = teacher_weights[item]

    student.load_state_dict(inherited_weights)


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
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--data-path', default='/fsx/laion/data/openprompts.csv', type=str, metavar='PATH',
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

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling sync BN.')
    args = parser.parse_args()
    return args


def main():
    prompt = "a photo of an astronaut riding a horse on mars"

    args = parse()

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

    # generator_t = torch.Generator("cuda").manual_seed(93)
    generator_s = torch.Generator("cuda").manual_seed(93)

    # pipe_teacher = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
    # pipe_student = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
    # pipe_teacher = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5")
    pipe = StableDiffusionPipelineAda.from_pretrained("./stable-diffusion-v1-5")

    noise_scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae

    # qing method
    # calc_SNR(noise_scheduler)

    # reconstruct student net for direct compact deployment #########################
    student_config = dict(pipe.unet.config)

    # student_config['layers_per_block'] = 1
    # student_config['block_out_channels'] = [k // 2 for k in student_config['block_out_channels']]
    # student_config['mid_block_type'] = None
    student_config['down_block_types'] = (
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    student_config['up_block_types'] = (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D"
    )

    # pipe.unet = UNet2DConditionModel(**OrderedDict(student_config))
    # setattr(pipe, 'unet_c', torch.nn.ModuleList([]))
    # for i in range(4):
    #     pipe.unet_c.append(UNet2DConditionModel(**OrderedDict(student_config)))
    setattr(pipe, 'unet_0', UNet2DConditionModel(**OrderedDict(student_config)))
    setattr(pipe, 'unet_1', UNet2DConditionModel(**OrderedDict(student_config)))
    setattr(pipe, 'unet_2', UNet2DConditionModel(**OrderedDict(student_config)))
    setattr(pipe, 'unet_3', UNet2DConditionModel(**OrderedDict(student_config)))

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.unet.eval()

    # load weights from teacher #######################################################

    # inherit_weights(pipe.unet, pipe_teacher.unet)

    # ##########################################################

    # pipe_teacher = pipe_teacher.to("cuda")
    pipe = pipe.to("cuda")
    # pipe.unet_c = pipe.unet_c.to('cuda')
    pipe.unet_0 = pipe.unet_0.to('cuda')
    pipe.unet_1 = pipe.unet_1.to('cuda')
    pipe.unet_2 = pipe.unet_2.to('cuda')
    pipe.unet_3 = pipe.unet_3.to('cuda')

    unet_0 = torch.nn.parallel.DistributedDataParallel(pipe.unet_0,
                                                       device_ids=[args.gpu],
                                                       broadcast_buffers=False)
    unet_1 = torch.nn.parallel.DistributedDataParallel(pipe.unet_1,
                                                       device_ids=[args.gpu],
                                                       broadcast_buffers=False)
    unet_2 = torch.nn.parallel.DistributedDataParallel(pipe.unet_2,
                                                       device_ids=[args.gpu],
                                                       broadcast_buffers=False)
    unet_3 = torch.nn.parallel.DistributedDataParallel(pipe.unet_3,
                                                       device_ids=[args.gpu],
                                                       broadcast_buffers=False)

    try:
        unet_0.load_state_dict(torch.load(os.path.join(args.output_path, 'student_unet.pth'),
                                          map_location='cpu'),
                               strict=True)
        unet_1.load_state_dict(torch.load(os.path.join(args.output_path, 'student_unet.pth'),
                                          map_location='cpu'),
                               strict=True)
        unet_2.load_state_dict(torch.load(os.path.join(args.output_path, 'student_unet.pth'),
                                          map_location='cpu'),
                               strict=True)
        unet_3.load_state_dict(torch.load(os.path.join(args.output_path, 'student_unet.pth'),
                                          map_location='cpu'),
                               strict=True)
        print('maybe restarting, loaded latest ckpt from current directory')
    except:
        print('maybe relaunch without any pretrain, dangerous!!!!!!!!!!!!!!')

    if args.resume:
        unet.load_state_dict(torch.load(args.resume, map_location='cpu'))

    # if args.local_rank == 0:
    #     print(unet)

    unet_0.train()
    unet_1.train()
    unet_2.train()
    unet_3.train()

    # irregular prune utilities ###############################################################################

    # prune utilities ###############################################################################

    # data and sampler ##################################################################################
    # open_prompt_data = CsvDataset(args.data_path) if args.dataset_type == 'csv' else TextDataset(args.data_path)
    #
    # train_sampler = None
    # # val_sampler = None
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(open_prompt_data)
    #     # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    #
    # dataloader = DataLoader(open_prompt_data, batch_size=args.batch_size,
    #                         shuffle=(train_sampler is None), num_workers=args.workers,
    #                         pin_memory=True, sampler=train_sampler)

    # ----------------------- Data Loading -----------------------
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # @mst: use wds dataset API
    class Args:
        pass  # to use open clip api

    data_args = Args()
    data_args.train_data = [os.path.join(args.data_path, x)
                            for x in os.listdir(args.data_path) if
                            x.endswith('.tar')]  # open('s3_urls.txt').read().splitlines()
    print(f'Found {len(data_args.train_data)} .tar files in {args.data_path}')
    data_args.train_num_samples = 2000000000
    data_args.train_data_upsampling_factors = None
    data_args.batch_size = args.batch_size
    data_args.world_size = torch.distributed.get_world_size()
    data_args.workers = args.workers
    data_args.seed = -1
    wds_dataset = get_wds_dataset(data_args,
                                  preprocess_img=train_transforms,
                                  is_train=True,
                                  epoch=0,
                                  floor=False,
                                  tokenizer=tokenizer)
    train_dataloader = wds_dataset.dataloader
    # data stuff end, deactivate if slow in debug ###############################################################

    # text drop for classifier free guidance
    # empty_prompt = torch.load('empty.pt', map_location='cuda')

    loss_meter = AverageMeter()
    criterion = torch.nn.MSELoss().cuda()
    # qing method
    # criterion = torch.nn.MSELoss(reduction='none').cuda()
    optimizer = optim.AdamW([
        # {'params': unet.parameters(), 'lr': args.lr},
        {'params': unet_0.parameters(), 'lr': args.lr},
        {'params': unet_1.parameters(), 'lr': args.lr},
        {'params': unet_2.parameters(), 'lr': args.lr},
        {'params': unet_3.parameters(), 'lr': args.lr},
        # {'params': pipe_student.vae.decoder.parameters(), 'lr': args.lr},
        # {'params': pipe_student.vae.post_quant_conv.parameters(), 'lr': args.lr},
    ])

    scaler = torch.cuda.amp.GradScaler()

    if args.local_rank == 0:
        os.makedirs(args.output_path, exist_ok=True)

    for epoch in range(args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            print('epoch: ', epoch)
        for mini_batch, batch in enumerate(train_dataloader):
            # Prepare input
            images, text_input_ids = batch
            images = images.cuda()

            text_input_ids = text_input_ids.squeeze(dim=1).cuda()  # [bs, 1, 77] --> [bs, 77]
            # text drop
            # drop_idx = torch.rand(text_input_ids.size(0), device='cuda') > 0.1
            # text_input_ids = torch.where(drop_idx.unsqueeze(1), text_input_ids, empty_prompt)

            batch = {
                'pixel_values': images,
                'input_ids': text_input_ids,
            }

            latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            # timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)

            for interval in range(4):
                timesteps = torch.randint(0, 250, (bsz,), device=latents.device) + (250 * interval)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    if interval == 0:
                        model_pred = unet_0(noisy_latents, timesteps, encoder_hidden_states).sample
                    elif interval == 1:
                        model_pred = unet_1(noisy_latents, timesteps, encoder_hidden_states).sample
                    elif interval == 2:
                        model_pred = unet_2(noisy_latents, timesteps, encoder_hidden_states).sample
                    elif interval == 3:
                        model_pred = unet_3(noisy_latents, timesteps, encoder_hidden_states).sample
                    else:
                        raise NotImplementedError

                    loss = criterion(model_pred.float(), target.float())
                    # qing method
                    # loss = criterion(model_pred.float(), target.float()).mean(dim=tuple(range(1, target.ndim)),
                    #                                                           keepdims=True)
                    # loss_weight = noisy_latents.SNR.clamp(max=5)
                    # loss = (loss * loss_weight).mean()

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_meter.update(loss.item())

            if mini_batch % 10 == 0 and args.local_rank == 0:
                print('iteration: ', mini_batch, 'current loss: ', loss_meter.avg)

            if mini_batch % 100 == 0 and args.local_rank == 0:
                unet_0.eval()
                unet_1.eval()
                unet_2.eval()
                unet_3.eval()
                image_student = pipe(prompt, num_inference_steps=50, generator=generator_s).images[0]
                image_student.save(
                    os.path.join(args.output_path, "astronaut_rides_horse_v15_student_epoch_" + str(epoch) +
                                 "_iter_" + str(mini_batch) + ".png"))
                # torch.save(unet.state_dict(), os.path.join(args.output_path, 'student_unet.pth'))
                unet_0.train()
                unet_1.train()
                unet_2.train()
                unet_3.train()

            if mini_batch % 1000 == 0 and args.local_rank == 0:
                # torch.save(unet.state_dict(), os.path.join(args.output_path,
                #                                            'student_unet_e_' + str(epoch) + '_iter_' + str(
                #                                                mini_batch) + '.pth'))
                torch.save(unet_0.state_dict(), os.path.join(args.output_path,
                                                             'student_unet0_e_' + str(epoch) + '_iter_' + str(
                                                                 mini_batch) + '.pth'))
                torch.save(unet_1.state_dict(), os.path.join(args.output_path,
                                                             'student_unet1_e_' + str(epoch) + '_iter_' + str(
                                                                 mini_batch) + '.pth'))
                torch.save(unet_2.state_dict(), os.path.join(args.output_path,
                                                             'student_unet2_e_' + str(epoch) + '_iter_' + str(
                                                                 mini_batch) + '.pth'))
                torch.save(unet_3.state_dict(), os.path.join(args.output_path,
                                                             'student_unet3_e_' + str(epoch) + '_iter_' + str(
                                                                 mini_batch) + '.pth'))
        loss_meter.reset()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
