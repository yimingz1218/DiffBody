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
    PNDMScheduler, LMSDiscreteScheduler, UniPCMultistepScheduler

from models.embedder import Embedder
from pipelines.pipeline_stable_diffusion_mb_downup import StableDiffusionPipeline
from collections import OrderedDict
import boto3
from diffusers.models.controlnet_composer import ControlNetModel
from pipelines.pipeline_controlnet_composer import StableDiffusionControlNetPipeline

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
    parser.add_argument("--pretrained_model_name_or_path",type=str,default=None,required=True)
    parser.add_argument("--pretrained_model_name_or_path2",type=str,default=None,required=True)
    parser.add_argument("--t2mn_path", type=str, default="/fsx_laion/alvin/sd4human/80g-rc-realvision-v-debug-glc-flaw-learn-w2in-noisyin-mb-copy2-downup-512/checkpoint-18000")
    parser.add_argument("--controlnet_model_name_or_path", type=str, default="/fsx_laion/alvin/sd4human/ctrl-realvision-eps-glc-composer-wmn-sum-512/checkpoint-50000")
    parser.add_argument('--prediction_type2', type=str, default='v_prediction', choices=['epsilon', 'v_prediction', 'target'], help='Select a mode')
    parser.add_argument("--cond_num", type=int, default=3)
    parser.add_argument('--fusion', type=str, default="sum")
    parser.add_argument("--validation_steps",type=int,default=500,)
    parser.add_argument("--test_data_dir",nargs='+',type=str,default=None,)
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
            "/fsx_laion/alvin/pretrain/sd-vae-ft-mse"
        )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    from diffusers.models.unet_2d_condition_multi_branch_downup import UNet2DConditionModel
    unet_t2mn = UNet2DConditionModel.from_pretrained(args.t2mn_path, subfolder="unet_ema")
    unet_t2mn.requires_grad_(False)
    
    unet = diffusers.UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path2, subfolder="unet", revision=args.revision
    )
    
    # if args.controlnet_model_name_or_path:
    logger.info("Loading existing controlnet weights")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, subfolder="controlnet")
    # else:
    #     logger.info("Initializing controlnet weights from unet")
    #     controlnet = ControlNetModel.from_unet(unet2, cond_num=args.cond_num, fusion=args.fusion, normalize_to_0_1=args.reserve_minus_one_to_one)
    # controlnet.config.cond_num = args.cond_num
    # controlnet.config["cond_num"] = args.cond_num
    # controlnet.config.fusion = args.fusion
    # controlnet.config["fusion"] = args.fusion
    # controlnet.config.normalize_to_0_1 = args.reserve_minus_one_to_one
    # controlnet.config["normalize_to_0_1"] = args.reserve_minus_one_to_one

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "controlnet"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    unet_t2mn.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    
    unet.eval()
    unet_t2mn.eval()
    controlnet.eval()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        unet_t2mn.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
        
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

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
            # data_args.val_data += [os.path.join(data_folder, x)
            #                         for x in os.listdir(data_folder) if
            #                         x.endswith('.tar')]  # open('s3_urls.txt').read().splitlines()
            data_list = [os.path.join(data_folder, x)
                                 for x in os.listdir(data_folder) if
                                 x.endswith('.tar')]  # open('s3_urls.txt').read().splitlines()
            data_list.sort()
            if "getty" in data_folder:
                data_list = data_list[:50]
            data_args.val_data += data_list
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
    unet, unet_t2mn, controlnet, test_dataloader = accelerator.prepare(
        unet, unet_t2mn, controlnet, test_dataloader
    )

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        # text_encoder=text_encoder,
        # tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet_t2mn),
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
    
    controlnet = accelerator.unwrap_model(controlnet)

    pipeline2 = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path2,
        vae=vae,
        # text_encoder=text_encoder,
        # tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline2.scheduler = UniPCMultistepScheduler.from_config(pipeline2.scheduler.config)
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
        
    step1 = args.t2mn_path.split('/')[-1].split("-")[1]
    save_path = os.path.join(args.output_dir, args.inference_folder_name, f"image-{step1}")
    os.makedirs(save_path, exist_ok=True)
    
    step2 = args.controlnet_model_name_or_path.split('/')[-1].split("-")[1]
    save_path2 = os.path.join(args.output_dir2, args.inference_folder_name2, f"image-{step2}")
    os.makedirs(save_path2, exist_ok=True)

    texts = []
    for step, batch in enumerate(test_dataloader):
        images, text_input_ids, text_raw, blip, blip_raw, \
                body, face, hand, normal, depth, midas_depth, canny, whole, descriptions, \
                    body_ori, face_ori, hand_ori, normal_ori, depth_ori, midas_depth_ori, canny_ori, whole_ori, \
                        body_dt, face_dt, hand_dt, normal_dt, depth_dt, midas_depth_dt, canny_dt, whole_dt = batch
                        
        images = images.to(unet.device)
        batch_size = images.shape[0]
        text_input_ids = text_input_ids.to(text_encoder.device)
        text_input_ids = text_input_ids.squeeze(dim=1)
        batch = {
            'pixel_values': images,
            'input_ids': text_input_ids,
            'text_raw': text_raw,
        }
        
        whole = whole.to(unet.device)
        batch["whole"] = whole
            
        for i_batch in range(batch_size):
            img_id = step * batch_size + i_batch
                
            with torch.autocast("cuda"):
                output = pipeline(
                    batch["text_raw"][i_batch], 
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=500, 
                    generator=generator,
                    batch=batch, 
                    args=args, 
                    guidance_rescale=0.7 if args.flaw else 0.,
                    negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                )

                image = output.images[0]
                image.save(os.path.join(save_path, f"{img_id}-rgb.png"))
                midas_depth_image = output.midas_depth_image[0]
                midas_depth_image.save(os.path.join(save_path, f"{img_id}-midas_depth.png"))
                normal_image = output.normal_image[0]
                normal_image.save(os.path.join(save_path, f"{img_id}-normal.png"))
            
            midas_depth_tensor = 2 * (transforms.ToTensor()(midas_depth_image)) - 1
            batch["midas_depth"] = midas_depth_tensor.unsqueeze(0).to(unet.device)
            normal_tensor = 2 * (transforms.ToTensor()(normal_image)) - 1
            batch["normal"] = normal_tensor.unsqueeze(0).to(unet.device)
            
            controlnet_image = []
            for key in ['depth', 'midas_depth', 'normal', 'canny', 'body', 'face', 'hand', 'whole']:
                if key in args.cond_type:
                    controlnet_image.append(batch[key][i_batch])
                    
            with torch.autocast("cuda"):
                output = pipeline2(
                    batch["text_raw"][i_batch], 
                    image=controlnet_image, 
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=500, 
                    generator=generator,
                    negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                )

                image = output.images[0]
                image.save(os.path.join(save_path2, f"{img_id}.png"))
            
        # del pipeline
        # torch.cuda.empty_cache()
        if step >= 15:
            break
    
    # with open(os.path.join(save_path, "text-full.txt"), 'w') as file:
    #     for item in texts:
    #         file.write(item + '\n')
                    
        # if args.use_ema:
        #     # Switch back to the original UNet parameters.
        #     ema_unet.restore(unet.parameters())


if __name__ == "__main__":
    main()
