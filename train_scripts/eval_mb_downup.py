#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from PIL import Image

import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from openclip.training.data import get_wds_dataset, get_wds_dataset_cond
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, \
    DEISMultistepScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, \
    PNDMScheduler, LMSDiscreteScheduler

from models.embedder import Embedder
from pipelines.pipeline_stable_diffusion_mb_downup import StableDiffusionPipeline
from collections import OrderedDict
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
import copy
import boto3
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
import json
import cv2
import seaborn as sns

if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    ################################### newly added args ###################################
    # parser.add_argument('--prediction_type', type=str, default='v_prediction', choices=['epsilon', 'v_prediction', 'target'], help='Select a mode')
    parser.add_argument('--inference_folder_name', type=str)
    parser.add_argument('--fusion', type=str, default="learn")
    parser.add_argument('--change_whole_to_body', default=False, action="store_true")
    parser.add_argument('--off_wa', default=False, action="store_true")
    parser.add_argument('--normalize_dist', default=False, action="store_true")
    parser.add_argument('--filter_wm', default=False, action="store_true")
    
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--rescale", type=float, default=0.75)
    
    # statistics for three datasets, laion+coyo+getty
    parser.add_argument("--rgb_mean", type=float, default=0.14654)
    parser.add_argument("--rgb_std", type=float, default=1.03744)
    # parser.add_argument("--whole_mean", type=float, default=0.14713)
    # parser.add_argument("--whole_std", type=float, default=0.96812)
    parser.add_argument("--whole_mean", type=float, default=-0.2599426086956522)
    parser.add_argument("--whole_std", type=float, default=1.3836632689065582)
    parser.add_argument("--body_mean", type=float, default=-0.2481)
    parser.add_argument("--body_std", type=float, default=1.45647)
    parser.add_argument("--depth_mean", type=float, default=0.21360)
    parser.add_argument("--depth_std", type=float, default=1.20629)
    parser.add_argument("--normal_mean", type=float, default=0.60303)
    parser.add_argument("--normal_std", type=float, default=0.91429)
    
    # # statistics for two datasetsm laion+coyo
    # parser.add_argument("--rgb_mean", type=float, default=0.144028)
    # parser.add_argument("--rgb_std", type=float, default=1.0420677550094796)
    # parser.add_argument("--whole_mean", type=float, default=-0.2598586666666667)
    # parser.add_argument("--whole_std", type=float, default=1.3824869261991977)
    # parser.add_argument("--body_mean", type=float, default=-0.2481)
    # parser.add_argument("--body_std", type=float, default=1.45647)
    # parser.add_argument("--depth_mean", type=float, default=0.22104533333333334)
    # parser.add_argument("--depth_std", type=float, default=1.2044201368629092)
    # parser.add_argument("--normal_mean", type=float, default=0.6173293333333333)
    # parser.add_argument("--normal_std", type=float, default=0.9108628719489077)
    
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument("--test_data_dir", nargs='+', type=str, default=None)
    parser.add_argument('--pred_null', default=False, action="store_true")
    parser.add_argument("--branch_num", type=int)
    parser.add_argument('--noisy_target', type=str, default=[], nargs="+", help='add which types of conditions')
    parser.add_argument('--noisy_cond', type=str, default=[], nargs="+", help='add which types of conditions')
    parser.add_argument('--cond_type', type=str, default=[], nargs="+", help='add which types of conditions')
    parser.add_argument("--copy_first_n_block", type=int)
    parser.add_argument("--copy_last_n_block", type=int)
    parser.add_argument("--timestep_start", type=int, default=0)
    parser.add_argument("--timestep_end", type=int, default=1000)
    parser.add_argument('--overfit', default=False, action="store_true")
    parser.add_argument('--rv_prompt', default=False, action="store_true")
    parser.add_argument('--copy_weight', default=False, action="store_true")
    parser.add_argument('--copy_weight_same', default=False, action="store_true")
    parser.add_argument('--target_change', type=str, choices=['depth', 'normal', 'canny', 'body', 'face', 'hand', 'whole'], help='how to inject the spatial condition')
    parser.add_argument('--size_cond', default=False, action="store_true")
    parser.add_argument('--flaw', default=False, action="store_true")
    parser.add_argument('--only_attn', default=False, action="store_true")
    parser.add_argument('--only_ca', default=False, action="store_true")
    parser.add_argument('--filter_mface', default=False, action="store_true")
    parser.add_argument('--filter_wpose', default=False, action="store_true")
    parser.add_argument('--filter_lowres', default=False, action="store_true")
    parser.add_argument("--filter_res", type=int)
    parser.add_argument("--validation_steps", type=int, default=500, help="Run validation every X epochs.")
    parser.add_argument('--dropout', default=False, action="store_true")
    parser.add_argument('--grid_dnc', default=False, action="store_true")
    parser.add_argument('--blip_concat', default=False, action="store_true")
    parser.add_argument('--string_concat', default=False, action="store_true")
    parser.add_argument('--string_substitute', default=False, action="store_true")
    parser.add_argument('--pred_cond', type=str, default=[], nargs="+", help='add which types of conditions')
    parser.add_argument("--depth_weight", type=float, default=0.1)
    parser.add_argument("--normal_weight", type=float, default=0.1)
    parser.add_argument("--canny_weight", type=float, default=0.1)
    parser.add_argument("--body_weight", type=float, default=0.1)
    parser.add_argument("--face_weight", type=float, default=0.1)
    parser.add_argument("--hand_weight", type=float, default=0.1)
    parser.add_argument("--whole_weight", type=float, default=0.1)
    parser.add_argument('--cond_reshape', type=str, choices=['resize', 'vae', 'learn_conv'], help='how to reshape the spatial condition to the same shape as the latent space size')
    parser.add_argument('--cond_inject', type=str, choices=['concat', 'spade', 'sum'], help='how to inject the spatial condition')
    parser.add_argument("--embedder_channel", default=4, type=int, help="channel number.")
    ################################### newly added args ###################################
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        nargs='+',
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

class CustomDataset(Dataset):
    def __init__(self, args, json_path='/fsx_laion/alvin/Dataset/coco/512x512-img_text_pose_val2014-human.json', image_processor=None):
        # self.depth = []
        self.normal = []
        self.midas_depth = []
        self.text = []
        self.id_list = []
        self.kp_list = []
        self.body_kp_list = []
        self.image_processor = image_processor
        
        self.args = args

        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
            
        keys_list = list(json_data.keys())
        keys_list.sort()
        key_to_inference = keys_list[self.args.start : self.args.end]
        
        for i, key in enumerate(key_to_inference):
            image_id = key
            content = json_data[key]
            # self.depth.append(f'/fsx_laion/alvin/Dataset/coco/512x512-val2014-human-structure/depth/COCO_val2014_{int(image_id):012d}.jpg')
            self.normal.append(f'/fsx_laion/alvin/Dataset/coco/512x512-val2014-human-structure/normal/COCO_val2014_{int(image_id):012d}.jpg')
            self.midas_depth.append(f'/fsx_laion/alvin/Dataset/coco/512x512-val2014-human-structure/midas_depth/COCO_val2014_{int(image_id):012d}.jpg')
            self.text.append(max(content['captions'], key=len))
            self.id_list.append(int(image_id))
            
            kp_coord = np.array(content["new_wholebody_kp"])
            kp_conf = np.array(content["new_wholebody_kp_score"])
            kp = np.concatenate([kp_coord, kp_conf[..., np.newaxis]], axis=-1)
            self.kp_list.append(kp)
            
            body_kp_coord = np.array(content["new_body_kp"])
            body_kp_conf = np.array(content["new_body_kp_score"])
            body_kp = np.concatenate([body_kp_coord, body_kp_conf[..., np.newaxis]], axis=-1)
            self.body_kp_list.append(body_kp)
            
    def __len__(self):
        return len(self.normal)
    
    def imshow_keypoints_whole(
        self,
        img,
        pose_result,
        skeleton=None,
        kpt_score_thr=0.3,
        pose_kpt_color=None,
        pose_link_color=None,
        radius=4,
        thickness=1,
        show_keypoint_weight=False,
        height=None,
        width=None):
        """Draw keypoints and links on an image.

        Args:
                img (str or Tensor): The image to draw poses on. If an image array
                    is given, id will be modified in-place.
                pose_result (list[kpts]): The poses to draw. Each element kpts is
                    a set of K keypoints as an Kx3 numpy.ndarray, where each
                    keypoint is represented as x, y, score.
                kpt_score_thr (float, optional): Minimum score of keypoints
                    to be shown. Default: 0.3.
                pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                    the keypoint will not be drawn.
                pose_link_color (np.array[Mx3]): Color of M links. If None, the
                    links will not be drawn.
                thickness (int): Thickness of lines.
        """

        # img = mmcv.imread(img)
        # img_h, img_w, _ = img.shape
        if img is None:
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img_h, img_w = height, width
        else:
            img_h, img_w, _ = img.shape

        for kpts in pose_result:

            kpts = np.array(kpts, copy=False)

            # draw each point on image
            if pose_kpt_color is not None:
                assert len(pose_kpt_color) == len(kpts)
                for kid, kpt in enumerate(kpts):
                    if kid in [17, 18, 19, 20, 21, 22]:
                        continue
                    if kid in [13, 14, 15, 16]:
                        if kpt[0] > min(kpts[23:91, 0]) and kpt[0] < max(kpts[23:91, 0]) and kpt[1] > min(kpts[23:91, 1]) and kpt[1] < max(kpts[23:91, 1]):
                            continue
                    x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
                    if kpt_score > kpt_score_thr:
                        color = tuple(int(c) for c in pose_kpt_color[kid])
                        if show_keypoint_weight:
                            img_copy = img.copy()
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                    radius, color, -1)
                            transparency = max(0, min(1, kpt_score))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)
                        else:
                            cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                                    color, -1)

            # draw links
            if skeleton is not None and pose_link_color is not None:
                assert len(pose_link_color) == len(skeleton)
                for sk_id, sk in enumerate(skeleton):
                    if sk[0] in [17, 18, 19, 20, 21, 22] or sk[1] in [17, 18, 19, 20, 21, 22]:
                        continue
                    if sk[0] in [13, 14, 15, 16]:
                        if kpts[sk[0], 0] > min(kpts[23:91, 0]) and kpts[sk[0], 0] < max(kpts[23:91, 0]) and kpts[sk[0], 1] > min(kpts[23:91, 1]) and kpts[sk[0], 1] < max(kpts[23:91, 1]):
                            continue
                    if sk[1] in [13, 14, 15, 16]:
                        if kpts[sk[1], 0] > min(kpts[23:91, 0]) and kpts[sk[1], 0] < max(kpts[23:91, 0]) and kpts[sk[1], 1] > min(kpts[23:91, 1]) and kpts[sk[1], 1] < max(kpts[23:91, 1]):
                            continue
                    pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                    pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                    # if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                    #         and pos1[1] < img_h and pos2[0] > 0 and pos2[0] < img_w
                    #         and pos2[1] > 0 and pos2[1] < img_h
                    #         and kpts[sk[0], 2] > kpt_score_thr
                    #         and kpts[sk[1], 2] > kpt_score_thr):
                    if (kpts[sk[0], 2] > kpt_score_thr
                            and kpts[sk[1], 2] > kpt_score_thr):
                        color = tuple(int(c) for c in pose_link_color[sk_id])
                        if show_keypoint_weight:
                            img_copy = img.copy()
                            X = (pos1[0], pos2[0])
                            Y = (pos1[1], pos2[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = thickness
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)), int(angle), 0,
                                360, 1)
                            cv2.fillConvexPoly(img_copy, polygon, color)
                            # transparency = max(
                            #     0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                            transparency = 1
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)
                        else:
                            cv2.line(img, pos1, pos2, color, thickness=thickness)

        return img
    
    def draw_whole_body_skeleton(
        self,
        img,
        pose,
        radius=4,
        thickness=1,
        kpt_score_thr=0.3,
        height=None,
        width=None,
        ):
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                                [230, 230, 0], [255, 153, 255], [153, 204, 255],
                                [255, 102, 255], [255, 51, 255], [102, 178, 255],
                                [51, 153, 255], [255, 153, 153], [255, 102, 102],
                                [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                [51, 255, 51], [0, 255, 0], [0, 0, 255],
                                [255, 0, 0], [255, 255, 255]])
        
        # below are for the whole body keypoints
        skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                            [8, 10], [1, 2], [0, 1], [0, 2],
                            [1, 3], [2, 4], [3, 5], [4, 6], [15, 17], [15, 18],
                            [15, 19], [16, 20], [16, 21], [16, 22], [91, 92],
                            [92, 93], [93, 94], [94, 95], [91, 96], [96, 97],
                            [97, 98], [98, 99], [91, 100], [100, 101], [101, 102],
                            [102, 103], [91, 104], [104, 105], [105, 106],
                            [106, 107], [91, 108], [108, 109], [109, 110],
                            [110, 111], [112, 113], [113, 114], [114, 115],
                            [115, 116], [112, 117], [117, 118], [118, 119],
                            [119, 120], [112, 121], [121, 122], [122, 123],
                            [123, 124], [112, 125], [125, 126], [126, 127],
                            [127, 128], [112, 129], [129, 130], [130, 131],
                            [131, 132]]

        pose_link_color = palette[[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ] + [16, 16, 16, 16, 16, 16] + [
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
            16
        ] + [
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
            16
        ]]
        pose_kpt_color = palette[
            [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0] +
            [0, 0, 0, 0, 0, 0] + [19] * (68 + 42)]
        
        draw = self.imshow_keypoints_whole(img, pose, skeleton, 
                        kpt_score_thr=0.3,
                        pose_kpt_color=pose_kpt_color,
                        pose_link_color=pose_link_color,
                        radius=radius,
                        thickness=thickness,
                        show_keypoint_weight=True,
                        height=height,
                        width=width)
        return draw
    
    def draw_humansd_skeleton(self, image, pose, mmpose_detection_thresh=0.3, height=None, width=None, humansd_skeleton_width=10):
        humansd_skeleton=[
                [0,0,1],
                [1,0,2],
                [2,1,3],
                [3,2,4],
                [4,3,5],
                [5,4,6],
                [6,5,7],
                [7,6,8],
                [8,7,9],
                [9,8,10],
                [10,5,11],
                [11,6,12],
                [12,11,13],
                [13,12,14],
                [14,13,15],
                [15,14,16],
            ]
        # humansd_skeleton_width=10
        humansd_color=sns.color_palette("hls", len(humansd_skeleton)) 
        
        def plot_kpts(img_draw, kpts, color, edgs,width):     
                for idx, kpta, kptb in edgs:
                    if kpts[kpta,2]>mmpose_detection_thresh and \
                        kpts[kptb,2]>mmpose_detection_thresh :
                        line_color = tuple([int(255*color_i) for color_i in color[idx]])
                        
                        cv2.line(img_draw, (int(kpts[kpta,0]),int(kpts[kpta,1])), (int(kpts[kptb,0]),int(kpts[kptb,1])), line_color,width)
                        cv2.circle(img_draw, (int(kpts[kpta,0]),int(kpts[kpta,1])), width//2, line_color, -1)
                        cv2.circle(img_draw, (int(kpts[kptb,0]),int(kpts[kptb,1])), width//2, line_color, -1)
        
        if image is None:
            pose_image = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            pose_image = np.array(image, dtype=np.uint8)
        for person_i in range(len(pose)):
            if np.sum(pose[person_i])>0:
                plot_kpts(pose_image, pose[person_i],humansd_color,humansd_skeleton,humansd_skeleton_width)
        
        return pose_image
    
    def __getitem__(self, idx):
        # depth_path = self.depth[idx]
        # depth_image = Image.open(depth_path).convert("RGB")  
        # resize = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)
        # depth_image = resize(depth_image)
        
        normal_path = self.normal[idx]
        normal_image = Image.open(normal_path).convert("RGB")  
        # resize = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)
        # normal_image = resize(normal_image)
        
        midas_depth_path = self.midas_depth[idx]
        midas_depth_image = Image.open(midas_depth_path).convert("RGB")  
        # resize = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)
        # midas_depth_image = resize(midas_depth_image)
        
        whole_all = self.kp_list[idx]
        
        whole_draw = self.draw_whole_body_skeleton(
            # img=np.array(sample["image"]),
            img=None,
            pose=whole_all,
            # radius=4, 
            # thickness=4,
            height=512,
            width=512,
        )
        whole_image = Image.fromarray(whole_draw)
        
        body_all = self.body_kp_list[idx]
        body_draw = self.draw_humansd_skeleton(
            # image=np.array(sample["image"]), 
            image=None,
            pose=body_all, 
            height=512, 
            width=512, 
            humansd_skeleton_width=10,
        )
        body_image = Image.fromarray(body_draw)
        
        preprocess = transforms.Compose(
            [
                transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(), 
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        normal = preprocess(normal_image)
        midas_depth = preprocess(midas_depth_image)
        whole = preprocess(whole_image)
        body = preprocess(body_image)
        
        if self.args.change_whole_to_body:
            whole = body
        
        # if self.image_processor:
        #     image = self.image_processor(images=image, return_tensors="pt")

        return normal, midas_depth, whole, self.text[idx], self.id_list[idx]
    
def collate_fn(batch):
    normal, midas_depth, whole, text, id = zip(*batch)
    normal = torch.stack(normal)
    midas_depth = torch.stack(midas_depth)
    whole = torch.stack(whole)
    
    return normal, midas_depth, whole, text, id


def main():
    args = parse_args()
    if args.change_whole_to_body:
        args.whole_mean = args.body_mean
        args.whole_std = args.body_std

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=18000))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # logging_dir=logging_dir,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if args.flaw:
        noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", rescale_betas_zero_snr=True, timestep_spacing="trailing")
        noise_scheduler.config.rescale_betas_zero_snr = True
        noise_scheduler.config['rescale_betas_zero_snr'] = True
        noise_scheduler.config.timestep_spacing = "trailing"
        noise_scheduler.config['timestep_spacing'] = "trailing"
    else:
        noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.config.prediction_type = args.prediction_type
    noise_scheduler.config['prediction_type'] = args.prediction_type
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        # vae = AutoencoderKL.from_pretrained(
        #     args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        # )
        vae = AutoencoderKL.from_pretrained(
            "/fsx_laion/alvin/pretrain/sd-vae-ft-mse"
        )
        
    from diffusers.models.unet_2d_condition_multi_branch_downup import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained_2d(args.pretrained_model_name_or_path, subfolder="unet", size_cond=args.size_cond, branch_num=args.branch_num, copy_first_n_block=args.copy_first_n_block, copy_last_n_block=args.copy_last_n_block, fusion=args.fusion, off_wa=args.off_wa)
    unet.config.fusion = args.fusion
    unet.config["fusion"] = args.fusion
    unet.config.size_cond = args.size_cond
    unet.config["size_cond"] = args.size_cond
    unet.config.off_wa = args.off_wa
    unet.config["off_wa"] = args.off_wa
    if args.size_cond:
        unet.config.addition_embed_type = "time"
        unet.config["addition_embed_type"] = "time"
        
    unet.config.in_channels = 8
    unet.config["in_channels"] = 8

    # Modify input layer to have additional structural condition channels
    weights = unet.conv_in.weight.clone()
    bias = unet.conv_in.bias.clone() 

    unet.conv_in = torch.nn.Conv2d(8, weights.shape[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    with torch.no_grad():
        unet.conv_in.weight[:, :4] = weights
        unet.conv_in.weight[:, 4:] = torch.zeros(unet.conv_in.weight[:, 4:].shape) # new weights initialized to zero
        # unet.conv_in.bias[:4] = torch.nn.Parameter(bias * 1. / (num_cond + 1))
        unet.conv_in.bias = torch.nn.Parameter(bias)
        
    for i in range(args.branch_num):
        unet.conv_in_branch[i] = copy.deepcopy(unet.conv_in)
        
    unet.requires_grad_(True)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        from diffusers.models.unet_2d_condition_multi_branch_downup import UNet2DConditionModel
        ema_unet = UNet2DConditionModel.from_pretrained_2d(args.pretrained_model_name_or_path, subfolder="unet", size_cond=args.size_cond, branch_num=args.branch_num, copy_first_n_block=args.copy_first_n_block, copy_last_n_block=args.copy_last_n_block, fusion=args.fusion, off_wa=args.off_wa)
        ema_unet.config.fusion = args.fusion
        ema_unet.config["fusion"] = args.fusion
        ema_unet.config.size_cond = args.size_cond
        ema_unet.config["size_cond"] = args.size_cond
        ema_unet.config.off_wa = args.off_wa
        ema_unet.config["off_wa"] = args.off_wa
        if args.size_cond:
            ema_unet.config.addition_embed_type = "time"
            ema_unet.config["addition_embed_type"] = "time"
            
        ema_unet.config.in_channels = 8
        ema_unet.config["in_channels"] = 8
        # Modify input layer to have additional structural condition channels
        weights = ema_unet.conv_in.weight.clone()
        bias = ema_unet.conv_in.bias.clone() 
        
        ema_unet.conv_in = torch.nn.Conv2d(8, weights.shape[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        with torch.no_grad():
            ema_unet.conv_in.weight[:, :4] = weights
            ema_unet.conv_in.weight[:, 4:] = torch.zeros(ema_unet.conv_in.weight[:, 4:].shape) # new weights initialized to zero
            # ema_unet.conv_in.bias[:4] = torch.nn.Parameter(bias * 1. / (num_cond + 1))
            ema_unet.conv_in.bias = torch.nn.Parameter(bias)
            
        for i in range(args.branch_num):
            ema_unet.conv_in_branch[i] = copy.deepcopy(ema_unet.conv_in)
            
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
        
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    if args.only_attn:
        for name, param in unet.named_parameters():
            if "transformer_blocks" not in name:
                param.requires_grad = False
                
    if args.only_ca:
        for name, param in unet.named_parameters():
            if "attn2" not in name:
                param.requires_grad = False
        
    # params_to_optimize = list(unet.parameters()) 
    params_to_optimize = [param for param in unet.parameters() if param.requires_grad]

    optimizer = optimizer_cls(
        # unet.parameters(),
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = CustomDataset(args)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_update_steps_per_epoch = math.ceil(400000000 / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, test_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, test_dataloader, lr_scheduler
    )
    
    # if args.cond_reshape == "learn_conv":
    #     if "depth" in args.cond_type:
    #         depth_embedder = accelerator.prepare(depth_embedder)
    #     if "normal" in args.cond_type:
    #         normal_embedder = accelerator.prepare(normal_embedder)
    #     if "canny" in args.cond_type:
    #         canny_embedder = accelerator.prepare(canny_embedder)

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_update_steps_per_epoch = math.ceil(400000000 / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            # path = os.path.basename(args.resume_from_checkpoint)
            path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.output_dir, path))
            accelerator.load_state(path)
            # global_step = int(path.split("-")[1])
            global_step = int(path.split("/")[-1].split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    if args.use_ema:
        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
        ema_unet.store(unet.parameters())
        ema_unet.copy_to(unet.parameters())
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.config.prediction_type = args.prediction_type
    pipeline.scheduler.config['prediction_type'] = args.prediction_type
    
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        
    save_path = os.path.join(args.inference_folder_name, f"image-{global_step}")
    os.makedirs(os.path.join(save_path, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'midas_depth'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'normal'), exist_ok=True)

    for i_test, batch in tqdm(enumerate(test_dataloader)):
        normal, midas_depth, whole, text, id = batch
        batch_size = normal.shape[0]
        batch = {}
        normal = normal.to(unet.device)
        batch["normal"] = normal

        midas_depth = midas_depth.to(unet.device)
        batch["midas_depth"] = midas_depth

        whole = whole.to(unet.device)
        batch["whole"] = whole
        
        for i_batch in range(batch_size):
            img_id = i_test * batch_size + i_batch
        
            with torch.autocast("cuda"):
                output = pipeline(
                    text[i_batch], 
                    num_inference_steps=50, 
                    height=args.resolution,
                    width=args.resolution,
                    generator=generator, 
                    batch=batch, 
                    args=args, 
                    # guidance_rescale=0.7 if args.flaw else 0.,
                    guidance_rescale=args.rescale,
                    original_size=(512, 512),
                    guidance_scale=args.cfg,
                )
                
            image = output.images[0]
            image.save(os.path.join(save_path, 'rgb', f"{int(id[i_batch]):012d}.jpg"))
            midas_depth_image = output.midas_depth_image[0]
            midas_depth_image.save(os.path.join(save_path, 'midas_depth', f"{int(id[i_batch]):012d}.jpg"))
            normal_image = output.normal_image[0]
            normal_image.save(os.path.join(save_path, 'normal', f"{int(id[i_batch]):012d}.jpg"))


if __name__ == "__main__":
    main()
