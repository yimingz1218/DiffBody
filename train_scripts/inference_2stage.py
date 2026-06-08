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
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from openclip.training.data import get_wds_dataset, get_wds_dataset_cond
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, \
    DEISMultistepScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, \
    PNDMScheduler, LMSDiscreteScheduler

from models.embedder import Embedder
from pipelines.pipeline_stable_diffusion_spade import StableDiffusionPipeline
from collections import OrderedDict
import boto3

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
    parser.add_argument('--prediction_type2', type=str, default='v_prediction', choices=['epsilon', 'v_prediction', 'target'], help='Select a mode')
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--test_data_dir",
        nargs='+',
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument('--flaw', default=False, action="store_true")
    parser.add_argument('--filter_lowres', default=False, action="store_true")
    parser.add_argument("--filter_res", type=int)
    parser.add_argument('--noisy_cond', type=str, default=[], nargs="+", help='add which types of conditions')
    parser.add_argument("--output_dir2", type=str, default="sd-model-finetuned")
    parser.add_argument('--cond_reshape2', type=str, choices=['resize', 'vae', 'learn_conv'], help='how to reshape the spatial condition to the same shape as the latent space size')
    parser.add_argument('--inference_folder_name2', type=str, help='how to reshape the spatial condition to the same shape as the latent space size')
    parser.add_argument('--cond_inject2', type=str, choices=['concat', 'spade', 'sum'], help='how to inject the spatial condition')
    parser.add_argument('--cond_type2', type=str, default=[], nargs="+", help='add which types of conditions')
    parser.add_argument('--cond_type_test2', type=str, default=None, nargs="+", help='add which types of conditions')
    parser.add_argument("--resume_from_checkpoint2", type=str, default=None)
    parser.add_argument('--pred_cond2', default=False, action="store_true")
    parser.add_argument('--save_cond2', default=False, action="store_true")
    parser.add_argument('--inference_folder_name', type=str, help='how to reshape the spatial condition to the same shape as the latent space size')
    parser.add_argument('--grid_dnc', default=False, action="store_true")
    parser.add_argument('--pred_cond', default=False, action="store_true")
    parser.add_argument('--save_cond', default=False, action="store_true")
    parser.add_argument('--cond_reshape', type=str, choices=['resize', 'vae', 'learn_conv'], help='how to reshape the spatial condition to the same shape as the latent space size')
    parser.add_argument('--cond_inject', type=str, choices=['concat', 'spade', 'sum'], help='how to inject the spatial condition')
    parser.add_argument('--cond_type', type=str, default=[], nargs="+", help='add which types of conditions')
    parser.add_argument('--cond_type_test', type=str, default=None, nargs="+", help='add which types of conditions')
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


def main():
    args = parse_args()

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

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # logging_dir=logging_dir,
        project_config=accelerator_project_config,
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
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
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
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
        
    if args.cond_inject == "spade":
        num_cond = 0
        if "depth" in args.cond_type: num_cond += 1
        if "normal" in args.cond_type: num_cond += 1
        if "canny" in args.cond_type: num_cond += 1
        if "body" in args.cond_type: num_cond += 1
        if "face" in args.cond_type: num_cond += 1
        if "hand" in args.cond_type: num_cond += 1
        label_channels = num_cond * 3
        from models.unet2d_spade import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained_2d(args.pretrained_model_name_or_path, subfolder="unet", label_channels=label_channels)
    else:
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.non_ema_revision
        )
        
        if args.cond_inject == "concat":
            num_per_cond_channel = 3 if args.cond_reshape == "resize" else 4
            num_cond = 0
            if "depth" in args.cond_type: num_cond += 1
            if "normal" in args.cond_type: num_cond += 1
            if "canny" in args.cond_type: num_cond += 1
            if "body" in args.cond_type: num_cond += 1
            if "face" in args.cond_type: num_cond += 1
            if "hand" in args.cond_type: num_cond += 1
            num_cond_channel = num_cond * num_per_cond_channel
        elif args.cond_inject == "sum":
            num_cond_channel = 3 if args.cond_reshape == "resize" else 4
            
        unet.config.in_channels = 4 + num_cond_channel
        unet.config["in_channels"] = 4 + num_cond_channel

        # Modify input layer to have additional structural condition channels
        weights = unet.conv_in.weight.clone()
        bias = unet.conv_in.bias.clone() 

        unet.conv_in = torch.nn.Conv2d(4 + num_cond_channel, weights.shape[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        with torch.no_grad():
            unet.conv_in.weight[:, :4] = weights # original weights
            unet.conv_in.weight[:, 4:] = torch.zeros(unet.conv_in.weight[:, 4:].shape) # new weights initialized to zero
            unet.conv_in.bias = torch.nn.Parameter(bias)
            
        if args.pred_cond:
            num_cond = 0
            if "depth" in args.cond_type: num_cond += 1
            if "normal" in args.cond_type: num_cond += 1
            if "canny" in args.cond_type: num_cond += 1
            if "body" in args.cond_type: num_cond += 1
            if "face" in args.cond_type: num_cond += 1
            if "hand" in args.cond_type: num_cond += 1
            unet.config.out_channels = 4 + num_cond * 4
            unet.config["out_channels"] = 4 + num_cond * 4

            # Modify input layer to have additional structural condition channels
            weights = unet.conv_out.weight.clone()
            bias = unet.conv_out.bias.clone()

            unet.conv_out = torch.nn.Conv2d(weights.shape[1], 4 + num_cond * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            with torch.no_grad():
                unet.conv_out.weight[:4] = weights # original weights
                unet.conv_out.weight[4:] = torch.zeros(unet.conv_out.weight[4:].shape) # new weights initialized to zero
                unet.conv_out.bias[:4] = torch.nn.Parameter(bias)
                unet.conv_out.bias[4:] = torch.zeros(unet.conv_out.bias[4:].shape)
                
    if args.cond_inject2 == "spade":
        num_cond = 0
        if "depth" in args.cond_type2: num_cond += 1
        if "normal" in args.cond_type2: num_cond += 1
        if "canny" in args.cond_type2: num_cond += 1
        if "body" in args.cond_type2: num_cond += 1
        if "face" in args.cond_type2: num_cond += 1
        if "hand" in args.cond_type2: num_cond += 1
        label_channels = num_cond * 3
        from models.unet2d_spade import UNet2DConditionModel
        unet2 = UNet2DConditionModel.from_pretrained_2d(args.pretrained_model_name_or_path, subfolder="unet", label_channels=label_channels)
    else:
        from diffusers import UNet2DConditionModel
        unet2 = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.non_ema_revision
        )
        
        if args.cond_inject2 == "concat":
            num_per_cond_channel = 3 if args.cond_reshape2 == "resize" else 4
            num_cond = 0
            if "depth" in args.cond_type2: num_cond += 1
            if "normal" in args.cond_type2: num_cond += 1
            if "canny" in args.cond_type2: num_cond += 1
            if "body" in args.cond_type2: num_cond += 1
            if "face" in args.cond_type2: num_cond += 1
            if "hand" in args.cond_type2: num_cond += 1
            num_cond_channel = num_cond * num_per_cond_channel
        elif args.cond_inject2 == "sum":
            num_cond_channel = 3 if args.cond_reshape2 == "resize" else 4
            
        unet2.config.in_channels = 4 + num_cond_channel
        unet2.config["in_channels"] = 4 + num_cond_channel

        # Modify input layer to have additional structural condition channels
        weights = unet2.conv_in.weight.clone()
        bias = unet2.conv_in.bias.clone() 

        unet2.conv_in = torch.nn.Conv2d(4 + num_cond_channel, weights.shape[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        with torch.no_grad():
            unet2.conv_in.weight[:, :4] = weights # original weights
            unet2.conv_in.weight[:, 4:] = torch.zeros(unet2.conv_in.weight[:, 4:].shape) # new weights initialized to zero
            unet2.conv_in.bias = torch.nn.Parameter(bias)
            
        if args.pred_cond2:
            num_cond = 0
            if "depth" in args.cond_type2: num_cond += 1
            if "normal" in args.cond_type2: num_cond += 1
            if "canny" in args.cond_type2: num_cond += 1
            if "body" in args.cond_type2: num_cond += 1
            if "face" in args.cond_type2: num_cond += 1
            if "hand" in args.cond_type2: num_cond += 1
            unet2.config.out_channels = 4 + num_cond * 4
            unet2.config["out_channels"] = 4 + num_cond * 4

            # Modify input layer to have additional structural condition channels
            weights = unet2.conv_out.weight.clone()
            bias = unet2.conv_out.bias.clone()

            unet2.conv_out = torch.nn.Conv2d(weights.shape[1], 4 + num_cond * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            with torch.no_grad():
                unet2.conv_out.weight[:4] = weights # original weights
                unet2.conv_out.weight[4:] = torch.zeros(unet2.conv_out.weight[4:].shape) # new weights initialized to zero
                unet2.conv_out.bias[:4] = torch.nn.Parameter(bias)
                unet2.conv_out.bias[4:] = torch.zeros(unet2.conv_out.bias[4:].shape)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        if args.cond_inject == "spade":
            num_cond = 0
            if "depth" in args.cond_type: num_cond += 1
            if "normal" in args.cond_type: num_cond += 1
            if "canny" in args.cond_type: num_cond += 1
            if "body" in args.cond_type: num_cond += 1
            if "face" in args.cond_type: num_cond += 1
            if "hand" in args.cond_type: num_cond += 1
            label_channels = num_cond * 3
            from models.unet2d_spade import UNet2DConditionModel
            ema_unet = UNet2DConditionModel.from_pretrained_2d(args.pretrained_model_name_or_path, subfolder="unet", label_channels=label_channels)
            ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
        else:
            from diffusers import UNet2DConditionModel
            ema_unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="unet",
                revision=args.non_ema_revision
            )
            
            if args.cond_inject == "concat":
                num_per_cond_channel = 3 if args.cond_reshape == "resize" else 4
                num_cond = 0
                if "depth" in args.cond_type: num_cond += 1
                if "normal" in args.cond_type: num_cond += 1
                if "canny" in args.cond_type: num_cond += 1
                if "body" in args.cond_type: num_cond += 1
                if "face" in args.cond_type: num_cond += 1
                if "hand" in args.cond_type: num_cond += 1
                num_cond_channel = num_cond * num_per_cond_channel
            elif args.cond_inject == "sum":
                num_cond_channel = 3 if args.cond_reshape == "resize" else 4
            ema_unet.config.in_channels = 4 + num_cond_channel
            ema_unet.config["in_channels"] = 4 + num_cond_channel
            # Modify input layer to have additional structural condition channels
            weights = ema_unet.conv_in.weight.clone()
            bias = ema_unet.conv_in.bias.clone() 
            ema_unet.conv_in = torch.nn.Conv2d(4 + num_cond_channel, weights.shape[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            with torch.no_grad():
                ema_unet.conv_in.weight[:, :4] = weights # original weights
                ema_unet.conv_in.weight[:, 4:] = torch.zeros(ema_unet.conv_in.weight[:, 4:].shape) # new weights initialized to zero
                ema_unet.conv_in.bias = torch.nn.Parameter(bias)
                
            ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
            
        if args.cond_inject2 == "spade":
            num_cond = 0
            if "depth" in args.cond_type2: num_cond += 1
            if "normal" in args.cond_type2: num_cond += 1
            if "canny" in args.cond_type2: num_cond += 1
            if "body" in args.cond_type2: num_cond += 1
            if "face" in args.cond_type2: num_cond += 1
            if "hand" in args.cond_type2: num_cond += 1
            label_channels = num_cond * 3
            from models.unet2d_spade import UNet2DConditionModel
            ema_unet2 = UNet2DConditionModel.from_pretrained_2d(args.pretrained_model_name_or_path, subfolder="unet", label_channels=label_channels)
            ema_unet2 = EMAModel(ema_unet2.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet2.config)
        else:
            from diffusers import UNet2DConditionModel
            ema_unet2 = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="unet",
                revision=args.non_ema_revision
            )
            
            if args.cond_inject2 == "concat":
                num_per_cond_channel = 3 if args.cond_reshape2 == "resize" else 4
                num_cond = 0
                if "depth" in args.cond_type2: num_cond += 1
                if "normal" in args.cond_type2: num_cond += 1
                if "canny" in args.cond_type2: num_cond += 1
                if "body" in args.cond_type2: num_cond += 1
                if "face" in args.cond_type2: num_cond += 1
                if "hand" in args.cond_type2: num_cond += 1
                num_cond_channel = num_cond * num_per_cond_channel
            elif args.cond_inject2 == "sum":
                num_cond_channel = 3 if args.cond_reshape2 == "resize" else 4
            ema_unet2.config.in_channels = 4 + num_cond_channel
            ema_unet2.config["in_channels"] = 4 + num_cond_channel
            # Modify input layer to have additional structural condition channels
            weights = ema_unet2.conv_in.weight.clone()
            bias = ema_unet2.conv_in.bias.clone() 
            ema_unet2.conv_in = torch.nn.Conv2d(4 + num_cond_channel, weights.shape[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            with torch.no_grad():
                ema_unet2.conv_in.weight[:, :4] = weights # original weights
                ema_unet2.conv_in.weight[:, 4:] = torch.zeros(ema_unet2.conv_in.weight[:, 4:].shape) # new weights initialized to zero
                ema_unet2.conv_in.bias = torch.nn.Parameter(bias)
                
            ema_unet2 = EMAModel(ema_unet2.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet2.config)

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
        
    params_to_optimize = list(unet.parameters()) 
    
    depth_embedder = normal_embedder = canny_embedder = body_embedder = face_embedder = hand_embedder = None
    
    if args.cond_reshape == "learn_conv":
        if "depth" in args.cond_type:
            depth_embedder = Embedder(conditioning_embedding_channels=args.embedder_channel, conditioning_channels=3).to(accelerator.device, dtype=weight_dtype)
            params_to_optimize.extend(list(depth_embedder.parameters()))
        if "normal" in args.cond_type:
            normal_embedder = Embedder(conditioning_embedding_channels=args.embedder_channel, conditioning_channels=3).to(accelerator.device, dtype=weight_dtype)
            params_to_optimize.extend(list(normal_embedder.parameters()))
        if "canny" in args.cond_type:
            canny_embedder = Embedder(conditioning_embedding_channels=args.embedder_channel, conditioning_channels=3).to(accelerator.device, dtype=weight_dtype)
            params_to_optimize.extend(list(canny_embedder.parameters()))
        if "body" in args.cond_type:
            body_embedder = Embedder(conditioning_embedding_channels=args.embedder_channel, conditioning_channels=3).to(accelerator.device, dtype=weight_dtype)
            params_to_optimize.extend(list(body_embedder.parameters()))
        if "face" in args.cond_type:
            face_embedder = Embedder(conditioning_embedding_channels=args.embedder_channel, conditioning_channels=3).to(accelerator.device, dtype=weight_dtype)
            params_to_optimize.extend(list(face_embedder.parameters()))
        if "hand" in args.cond_type:
            hand_embedder = Embedder(conditioning_embedding_channels=args.embedder_channel, conditioning_channels=3).to(accelerator.device, dtype=weight_dtype)
            params_to_optimize.extend(list(hand_embedder.parameters()))

    optimizer = optimizer_cls(
        # unet.parameters(),
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # # Preprocessing the datasets.
    # train_transforms = transforms.Compose(
    #     [
    #         transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    #         transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
    #         transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5]),
    #     ]
    # )

    class Args:
        pass  # to use open clip api

    data_args = Args()
    data_args.val_data = []
    try:
        from urlparse import urlparse
    except ImportError:
        from urllib.parse import urlparse


    class S3Url(object):
        """
        >>> s = S3Url("s3://bucket/hello/world")
        >>> s.bucket
        'bucket'
        >>> s.key
        'hello/world'
        >>> s.url
        's3://bucket/hello/world'
        >>> s = S3Url("s3://bucket/hello/world?qwe1=3#ddd")
        >>> s.bucket
        'bucket'
        >>> s.key
        'hello/world?qwe1=3#ddd'
        >>> s.url
        's3://bucket/hello/world?qwe1=3#ddd'
        >>> s = S3Url("s3://bucket/hello/world#foo?bar=2")
        >>> s.key
        'hello/world#foo?bar=2'
        >>> s.url
        's3://bucket/hello/world#foo?bar=2'
        """

        def __init__(self, url):
            self._parsed = urlparse(url, allow_fragments=False)

        @property
        def bucket(self):
            return self._parsed.netloc

        @property
        def key(self):
            if self._parsed.query:
                return self._parsed.path.lstrip('/') + '?' + self._parsed.query
            else:
                return self._parsed.path.lstrip('/')

        @property
        def url(self):
            return self._parsed.geturl()

    for data_folder in args.test_data_dir:
        
        # read from s3
        if data_folder.startswith('s3://'):
            s = S3Url(data_folder)
            s3 = boto3.resource('s3')
            my_bucket = s3.Bucket(s.bucket)

            for object_summary in my_bucket.objects.filter(Prefix=s.key):
                if object_summary.key.endswith(".tar"):
                    data_args.val_data.append(f'pipe:aws s3 cp s3://{s.bucket}/{object_summary.key} -')
        
        # read from fsx
        else:
            data_args.val_data += [os.path.join(data_folder, x)
                                    for x in os.listdir(data_folder) if
                                    x.endswith('.tar')]  # open('s3_urls.txt').read().splitlines()
    # for data_folder in args.test_data_dir:
    #     data_args.val_data += [os.path.join(data_folder, x)
    #                              for x in os.listdir(data_folder) if
    #                              x.endswith('.tar')]  # open('s3_urls.txt').read().splitlines()

    print(f'Found {len(data_args.val_data)} .tar files in {args.test_data_dir}')
    # data_args.train_num_samples = 400000000
    data_args.train_data_upsampling_factors = None
    data_args.val_num_samples = 64
    data_args.batch_size = 1
    # data_args.world_size = torch.distributed.get_world_size()
    # print(torch.distributed.get_world_size())
    data_args.workers = args.dataloader_num_workers
    data_args.seed = -1
    test_dataset = get_wds_dataset_cond(data_args,
                                  main_args=args,
                                  is_train=False,
                                  epoch=0,
                                  floor=False,
                                  tokenizer=tokenizer,
                                  dropout=False,
                                  grid_dnc=args.grid_dnc,
                                  filter_lowres=args.filter_lowres,
                                  )
    # train_dataset = train_dataset.with_length(300000000)
    test_dataloader = test_dataset.dataloader

    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    #     num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    # )

    # # Prepare everything with our `accelerator`.
    # unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     unet, optimizer, train_dataloader, lr_scheduler
    # )

    if args.use_ema:
        ema_unet.to(accelerator.device)
        ema_unet2.to(accelerator.device)

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
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
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            
            if args.cond_reshape == "learn_conv":
                if "depth" in args.cond_type:
                    depth_state_dict = torch.load(os.path.join(args.output_dir, path, "depth_embedder.pth"))
                    new_state_dict = OrderedDict()
                    for k, v in depth_state_dict.items():
                        name = k[7:] if k[:6] == 'module' else k 
                        new_state_dict[name] = v
                    depth_embedder.load_state_dict(new_state_dict)
                if "normal" in args.cond_type:
                    normal_state_dict = torch.load(os.path.join(args.output_dir, path, "normal_embedder.pth"))
                    new_state_dict = OrderedDict()
                    for k, v in normal_state_dict.items():
                        name = k[7:] if k[:6] == 'module' else k 
                        new_state_dict[name] = v
                    normal_embedder.load_state_dict(new_state_dict)
                if "canny" in args.cond_type:
                    canny_state_dict = torch.load(os.path.join(args.output_dir, path, "canny_embedder.pth"))
                    new_state_dict = OrderedDict()
                    for k, v in canny_state_dict.items():
                        name = k[7:] if k[:6] == 'module' else k 
                        new_state_dict[name] = v
                    canny_embedder.load_state_dict(new_state_dict)
                if "body" in args.cond_type:
                    body_state_dict = torch.load(os.path.join(args.output_dir, path, "body_embedder.pth"))
                    new_state_dict = OrderedDict()
                    for k, v in body_state_dict.items():
                        name = k[7:] if k[:6] == 'module' else k 
                        new_state_dict[name] = v
                    body_embedder.load_state_dict(new_state_dict)
                if "face" in args.cond_type:
                    face_state_dict = torch.load(os.path.join(args.output_dir, path, "face_embedder.pth"))
                    new_state_dict = OrderedDict()
                    for k, v in face_state_dict.items():
                        name = k[7:] if k[:6] == 'module' else k 
                        new_state_dict[name] = v
                    face_embedder.load_state_dict(new_state_dict)
                if "hand" in args.cond_type:
                    hand_state_dict = torch.load(os.path.join(args.output_dir, path, "hand_embedder.pth"))
                    new_state_dict = OrderedDict()
                    for k, v in hand_state_dict.items():
                        name = k[7:] if k[:6] == 'module' else k 
                        new_state_dict[name] = v
                    hand_embedder.load_state_dict(new_state_dict)
                    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint2:
        if args.resume_from_checkpoint2 != "latest":
            path = os.path.basename(args.resume_from_checkpoint2)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir2)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint2}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint2 = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(args.resume_from_checkpoint2, "unet_ema"), UNet2DConditionModel)
                ema_unet2.load_state_dict(load_model.state_dict())
                ema_unet2.to(accelerator.device)
                del load_model
            # models = unet2
            # for i in range(len(models)):
            #     # pop models so that they are not loaded again
            #     model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(args.resume_from_checkpoint2, subfolder="unet")
            unet2.register_to_config(**load_model.config)

            unet2.load_state_dict(load_model.state_dict())
            del load_model
            # stage2_state_dict = torch.load(os.path.join(args.output_dir2, path, "depth_embedder.pth"))
            # accelerator.load_state(os.path.join(args.output_dir2, path))
            global_step2 = int(path.split("-")[1])

            resume_global_step2 = global_step2 * args.gradient_accumulation_steps
            
            if args.cond_reshape == "learn_conv":
                if "depth" in args.cond_type:
                    depth_state_dict = torch.load(os.path.join(args.output_dir2, path, "depth_embedder.pth"))
                    new_state_dict = OrderedDict()
                    for k, v in depth_state_dict.items():
                        name = k[7:] if k[:6] == 'module' else k 
                        new_state_dict[name] = v
                    depth_embedder.load_state_dict(new_state_dict)
                if "normal" in args.cond_type:
                    normal_state_dict = torch.load(os.path.join(args.output_dir2, path, "normal_embedder.pth"))
                    new_state_dict = OrderedDict()
                    for k, v in normal_state_dict.items():
                        name = k[7:] if k[:6] == 'module' else k 
                        new_state_dict[name] = v
                    normal_embedder.load_state_dict(new_state_dict)
                if "canny" in args.cond_type:
                    canny_state_dict = torch.load(os.path.join(args.output_dir2, path, "canny_embedder.pth"))
                    new_state_dict = OrderedDict()
                    for k, v in canny_state_dict.items():
                        name = k[7:] if k[:6] == 'module' else k 
                        new_state_dict[name] = v
                    canny_embedder.load_state_dict(new_state_dict)
                if "body" in args.cond_type:
                    body_state_dict = torch.load(os.path.join(args.output_dir2, path, "body_embedder.pth"))
                    new_state_dict = OrderedDict()
                    for k, v in body_state_dict.items():
                        name = k[7:] if k[:6] == 'module' else k 
                        new_state_dict[name] = v
                    body_embedder.load_state_dict(new_state_dict)
                if "face" in args.cond_type:
                    face_state_dict = torch.load(os.path.join(args.output_dir2, path, "face_embedder.pth"))
                    new_state_dict = OrderedDict()
                    for k, v in face_state_dict.items():
                        name = k[7:] if k[:6] == 'module' else k 
                        new_state_dict[name] = v
                    face_embedder.load_state_dict(new_state_dict)
                if "hand" in args.cond_type:
                    hand_state_dict = torch.load(os.path.join(args.output_dir2, path, "hand_embedder.pth"))
                    new_state_dict = OrderedDict()
                    for k, v in hand_state_dict.items():
                        name = k[7:] if k[:6] == 'module' else k 
                        new_state_dict[name] = v
                    hand_embedder.load_state_dict(new_state_dict)

    # # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    # progress_bar.set_description("Steps")
    
    if args.use_ema:
        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
        ema_unet.store(unet.parameters())
        ema_unet.copy_to(unet.parameters())
        
        ema_unet2.store(unet2.parameters())
        ema_unet2.copy_to(unet2.parameters())
    
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
    if args.flaw:
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")
        pipeline.scheduler.config.rescale_betas_zero_snr = True
        pipeline.scheduler.config['rescale_betas_zero_snr'] = True
        pipeline.scheduler.config.timestep_spacing = "trailing"
        pipeline.scheduler.config['timestep_spacing'] = "trailing"
    else:
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(500)
    
    pipeline.scheduler.config.prediction_type = args.prediction_type
    pipeline.scheduler.config['prediction_type'] = args.prediction_type
    
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=False)
    
    pipeline2 = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet2),
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    if args.flaw:
        pipeline2.scheduler = DDIMScheduler.from_config(pipeline2.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")
        pipeline2.scheduler.config.rescale_betas_zero_snr = True
        pipeline2.scheduler.config['rescale_betas_zero_snr'] = True
        pipeline2.scheduler.config.timestep_spacing = "trailing"
        pipeline2.scheduler.config['timestep_spacing'] = "trailing"
    else:
        pipeline2.scheduler = DDIMScheduler.from_config(pipeline2.scheduler.config)
    pipeline2.scheduler.set_timesteps(500)
    
    pipeline2.scheduler.config.prediction_type = args.prediction_type2
    pipeline2.scheduler.config['prediction_type'] = args.prediction_type2
    
    pipeline2 = pipeline2.to(accelerator.device)
    pipeline2.set_progress_bar_config(disable=False)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline2.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        
    save_path = os.path.join(args.output_dir, args.inference_folder_name, f"image-{global_step}")
    os.makedirs(save_path, exist_ok=True)
    
    save_path2 = os.path.join(args.output_dir2, args.inference_folder_name2, f"image-{global_step2}")
    os.makedirs(save_path2, exist_ok=True)

    texts = []
    for step, batch in enumerate(test_dataloader):
        # try:
        # images, text_input_ids = batch
        images, text_input_ids, text_raw, blip, blip_raw, body, face, hand, normal, depth, canny, description = batch
        # images, text_input_ids, blip, normal, depth, canny = batch
        images = images.to(unet.device)
        batch_size = images.shape[0]
        text_input_ids = text_input_ids.to(text_encoder.device)
        text_input_ids = text_input_ids.squeeze(dim=1)
        batch = {
            'pixel_values': images,
            'input_ids': text_input_ids,
            'text_raw': text_raw,
        }
        
        if "blip" in args.cond_type:
            blip = blip.to(text_encoder.device)
            blip = blip.squeeze(dim=1)
            batch["blip"] = blip
            batch["blip_raw"] = blip_raw
        if "normal" in args.cond_type:
            normal = normal.to(unet.device)
            batch["normal"] = normal
        if "depth" in args.cond_type:
            depth = depth.to(unet.device)
            batch["depth"] = depth
        if "canny" in args.cond_type:
            canny = canny.to(unet.device)
            batch["canny"] = canny
        if "body" in args.cond_type:
            body = body.to(unet.device)
            batch["body"] = body
        if "face" in args.cond_type:
            face = face.to(unet.device)
            batch["face"] = face
        if "hand" in args.cond_type:
            hand = hand.to(unet.device)
            batch["hand"] = hand

        for i_batch in range(batch_size):
            img_id = step * batch_size + i_batch
            image_denormalize = (batch["pixel_values"] + 1) / 2.0
            image_numpy = image_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
            image_numpy = (image_numpy * 255).round().astype("uint8")
            image_pil = Image.fromarray(image_numpy)
            if args.save_cond:
                image_pil.save(os.path.join(save_path, f"image-{img_id}.png"))
            if "depth" in args.cond_type:
                # structural_cond.append(batch["depth"])
                # save the condition images
                depth_denormalize = (batch["depth"] + 1) / 2.0
                depth_numpy = depth_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                depth_numpy = (depth_numpy * 255).round().astype("uint8")
                depth_pil = Image.fromarray(depth_numpy)
                if args.save_cond:
                    depth_pil.save(os.path.join(save_path, f"depth-{img_id}.png"))
                if "depth" in args.cond_type_test:
                    batch["depth"] = batch["depth"][0].unsqueeze(0)
                else:
                    batch["depth"] = (torch.ones_like(batch["depth"][0].unsqueeze(0)) * (-1)).to(unet.device)
            if "normal" in args.cond_type:
                # structural_cond.append(batch["normal"])
                normal_denormalize = (batch["normal"] + 1) / 2.0
                normal_numpy = normal_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                normal_numpy = (normal_numpy * 255).round().astype("uint8")
                normal_pil = Image.fromarray(normal_numpy)
                if args.save_cond:
                    normal_pil.save(os.path.join(save_path, f"normal-{img_id}.png"))
                if "normal" in args.cond_type_test:
                    batch["normal"] = batch["normal"][0].unsqueeze(0)
                else:
                    batch["normal"] = (torch.ones_like(batch["normal"][0].unsqueeze(0)) * (-1)).to(unet.device)
            if "canny" in args.cond_type:
                # structural_cond.append(batch["canny"])
                canny_denormalize = (batch["canny"] + 1) / 2.0
                canny_numpy = canny_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                canny_numpy = (canny_numpy * 255).round().astype("uint8")
                canny_pil = Image.fromarray(canny_numpy)
                if args.save_cond:
                    canny_pil.save(os.path.join(save_path, f"canny-{img_id}.png"))
                if "canny" in args.cond_type_test:
                    batch["canny"] = batch["canny"][0].unsqueeze(0)
                else:
                    batch["canny"] = (torch.ones_like(batch["canny"][0].unsqueeze(0)) * (-1)).to(unet.device)
            if "body" in args.cond_type:
                # structural_cond.append(batch["body"])
                body_denormalize = (batch["body"] + 1) / 2.0
                body_numpy = body_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                body_numpy = (body_numpy * 255).round().astype("uint8")
                body_pil = Image.fromarray(body_numpy)
                if args.save_cond:
                    body_pil.save(os.path.join(save_path, f"body-{img_id}.png"))
                if "body" in args.cond_type_test:
                    batch["body"] = batch["body"][0].unsqueeze(0)
                else:
                    batch["body"] = (torch.ones_like(batch["body"][0].unsqueeze(0)) * (-1)).to(unet.device)
            if "face" in args.cond_type:
                # structural_cond.append(batch["face"])
                face_denormalize = (batch["face"] + 1) / 2.0
                face_numpy = face_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                face_numpy = (face_numpy * 255).round().astype("uint8")
                face_pil = Image.fromarray(face_numpy)
                if args.save_cond:
                    face_pil.save(os.path.join(save_path, f"face-{img_id}.png"))
                if "face" in args.cond_type_test:
                    batch["face"] = batch["face"][0].unsqueeze(0)
                else:
                    batch["face"] = (torch.ones_like(batch["face"][0].unsqueeze(0)) * (-1)).to(unet.device)
            if "hand" in args.cond_type:
                # structural_cond.append(batch["hand"])
                hand_denormalize = (batch["hand"] + 1) / 2.0
                hand_numpy = hand_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                hand_numpy = (hand_numpy * 255).round().astype("uint8")
                hand_pil = Image.fromarray(hand_numpy)
                if args.save_cond:
                    hand_pil.save(os.path.join(save_path, f"hand-{img_id}.png"))
                if "hand" in args.cond_type_test:
                    batch["hand"] = batch["hand"][0].unsqueeze(0)
                else:
                    batch["hand"] = (torch.ones_like(batch["hand"][0].unsqueeze(0)) * (-1)).to(unet.device)
            
            # for i in range(len(args.validation_prompts)):
            with torch.autocast("cuda"):
                image = pipeline(
                    # args.validation_prompts[i], 
                    batch["text_raw"][i_batch],
                    num_inference_steps=500, 
                    generator=generator, 
                    batch=batch, 
                    args=args, 
                    negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                    depth_embedder=depth_embedder, 
                    normal_embedder=normal_embedder,
                    canny_embedder=canny_embedder,
                    body_embedder=body_embedder, 
                    face_embedder=face_embedder,
                    hand_embedder=hand_embedder,
                    # guidance_rescale=0.7 if args.flaw else 0.,
                ).images[0]
            image.save(os.path.join(save_path, f"{img_id}.png"))
            # texts.append(batch["text_raw"][i_batch])
            
            image_tensor = transforms.ToTensor()(image)
            batch["depth"] = image_tensor.unsqueeze(0).to(unet.device)
            # batch["depth"] = image_tensor[:, :args.resolution // 2, args.resolution // 2:].unsqueeze(0)
            # batch["depth"] = F.interpolate(batch["depth"], size=(512, 512), mode='bilinear', align_corners=False).to(unet.device)
            # batch["normal"] = image_tensor[:, args.resolution // 2:, :args.resolution // 2].float().unsqueeze(0)
            # batch["normal"] = F.interpolate(batch["normal"], size=(512, 512), mode='bilinear', align_corners=False).to(unet.device)
            # batch["canny"] = image_tensor[:, args.resolution // 2:, args.resolution // 2:].float().unsqueeze(0)
            # batch["canny"] = F.interpolate(batch["canny"], size=(512, 512), mode='bilinear', align_corners=False).to(unet.device)
            # batch["body"] = batch["face"] = batch["hand"] = torch.zeros((1, 3, 512, 512))
            print(batch["depth"].shape)
            if "depth" in args.cond_type2:
                # structural_cond.append(batch["depth"])
                # save the condition images
                depth_denormalize = (batch["depth"] + 1) / 2.0
                depth_numpy = depth_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                depth_numpy = (depth_numpy * 255).round().astype("uint8")
                depth_pil = Image.fromarray(depth_numpy)
                if args.save_cond2:
                    depth_pil.save(os.path.join(save_path2, f"depth-{img_id}.png"))
                if "depth" in args.cond_type_test2:
                    batch["depth"] = batch["depth"][0].unsqueeze(0)
                else:
                    batch["depth"] = (torch.ones_like(batch["depth"][0].unsqueeze(0)) * (-1)).to(unet2.device)
            if "normal" in args.cond_type2:
                # structural_cond.append(batch["normal"])
                normal_denormalize = (batch["normal"] + 1) / 2.0
                normal_numpy = normal_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                normal_numpy = (normal_numpy * 255).round().astype("uint8")
                normal_pil = Image.fromarray(normal_numpy)
                if args.save_cond2:
                    normal_pil.save(os.path.join(save_path2, f"normal-{img_id}.png"))
                if "normal" in args.cond_type_test2:
                    batch["normal"] = batch["normal"][0].unsqueeze(0)
                else:
                    batch["normal"] = (torch.ones_like(batch["normal"][0].unsqueeze(0)) * (-1)).to(unet2.device)
            if "canny" in args.cond_type2:
                # structural_cond.append(batch["canny"])
                canny_denormalize = (batch["canny"] + 1) / 2.0
                canny_numpy = canny_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                canny_numpy = (canny_numpy * 255).round().astype("uint8")
                canny_pil = Image.fromarray(canny_numpy)
                if args.save_cond2:
                    canny_pil.save(os.path.join(save_path2, f"canny-{img_id}.png"))
                if "canny" in args.cond_type_test2:
                    batch["canny"] = batch["canny"][0].unsqueeze(0)
                else:
                    batch["canny"] = (torch.ones_like(batch["canny"][0].unsqueeze(0)) * (-1)).to(unet2.device)
            if "body" in args.cond_type2:
                # structural_cond.append(batch["body"])
                body_denormalize = (batch["body"] + 1) / 2.0
                body_numpy = body_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                body_numpy = (body_numpy * 255).round().astype("uint8")
                body_pil = Image.fromarray(body_numpy)
                if args.save_cond2:
                    body_pil.save(os.path.join(save_path2, f"body-{img_id}.png"))
                if "body" in args.cond_type_test2:
                    batch["body"] = batch["body"][0].unsqueeze(0)
                else:
                    batch["body"] = (torch.ones_like(batch["body"][0].unsqueeze(0)) * (-1)).to(unet2.device)
            if "face" in args.cond_type2:
                # structural_cond.append(batch["face"])
                face_denormalize = (batch["face"] + 1) / 2.0
                face_numpy = face_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                face_numpy = (face_numpy * 255).round().astype("uint8")
                face_pil = Image.fromarray(face_numpy)
                if args.save_cond2:
                    face_pil.save(os.path.join(save_path2, f"face-{img_id}.png"))
                if "face" in args.cond_type_test2:
                    batch["face"] = batch["face"][0].unsqueeze(0)
                else:
                    batch["face"] = (torch.ones_like(batch["face"][0].unsqueeze(0)) * (-1)).to(unet2.device)
            if "hand" in args.cond_type2:
                # structural_cond.append(batch["hand"])
                hand_denormalize = (batch["hand"] + 1) / 2.0
                hand_numpy = hand_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                hand_numpy = (hand_numpy * 255).round().astype("uint8")
                hand_pil = Image.fromarray(hand_numpy)
                if args.save_cond2:
                    hand_pil.save(os.path.join(save_path2, f"hand-{img_id}.png"))
                if "hand" in args.cond_type_test2:
                    batch["hand"] = batch["hand"][0].unsqueeze(0)
                else:
                    batch["hand"] = (torch.ones_like(batch["hand"][0].unsqueeze(0)) * (-1)).to(unet2.device)

            cond_inject = args.cond_inject
            cond_reshape = args.cond_reshape
            cond_type = args.cond_type
            args.cond_inject = args.cond_inject2
            args.cond_reshape = args.cond_reshape2
            args.cond_type = args.cond_type2
            
            with torch.autocast("cuda"):
                image = pipeline2(
                    # args.validation_prompts[i], 
                    batch["text_raw"][i_batch],
                    num_inference_steps=500, 
                    generator=generator, 
                    batch=batch, 
                    args=args, 
                    negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                    depth_embedder=depth_embedder, 
                    normal_embedder=normal_embedder,
                    canny_embedder=canny_embedder,
                    body_embedder=body_embedder, 
                    face_embedder=face_embedder,
                    hand_embedder=hand_embedder,
                    # guidance_rescale=0.7 if args.flaw else 0.,
                ).images[0]
            image.save(os.path.join(save_path2, f"{img_id}.png"))
            # texts.append(batch["text_raw"][i_batch])
            
            args.cond_inject = cond_inject
            args.cond_reshape = cond_reshape 
            args.cond_type = cond_type
            
        # del pipeline
        # torch.cuda.empty_cache()
        if step >= 20:
            break
    
    # with open(os.path.join(save_path, "text-full.txt"), 'w') as file:
    #     for item in texts:
    #         file.write(item + '\n')
                    
        # if args.use_ema:
        #     # Switch back to the original UNet parameters.
        #     ema_unet.restore(unet.parameters())


if __name__ == "__main__":
    main()
