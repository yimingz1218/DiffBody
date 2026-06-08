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
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from transformers.utils import ContextManagers
from PIL import Image

import diffusers
from diffusers import AutoencoderKL, IFPipeline
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
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, global_step, batch=None, depth_embedder=None, normal_embedder=None, canny_embedder=None, body_embedder=None, face_embedder=None, hand_embedder=None):
    logger.info("Running validation... ")

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
    pipeline.scheduler.set_timesteps(50)
    # if global_step != 0:
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

    save_path = os.path.join(args.output_dir, "images", f"image-{global_step}")
    os.makedirs(save_path, exist_ok=True)
    # print(len(args.validation_prompts))
    # if args.cond_inject == "spade":
    if batch is not None:
    #     num_cond = 0
    #     if "depth" in args.cond_type: num_cond += 1
    #     if "normal" in args.cond_type: num_cond += 1
    #     if "canny" in args.cond_type: num_cond += 1
    #     if "body" in args.cond_type: num_cond += 1
    #     if "face" in args.cond_type: num_cond += 1
    #     if "hand" in args.cond_type: num_cond += 1
    #     structural_cond = torch.zeros((1, 3 * num_cond, args.resolution, args.resolution)).to(unet.device)
    # else:
        # structural_cond = []
        image_denormalize = (batch["pixel_values"] + 1) / 2.0
        image_numpy = image_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        image_numpy = (image_numpy * 255).round().astype("uint8")
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(os.path.join(save_path, f"image.png"))
        if "depth" in args.cond_type:
            # structural_cond.append(batch["depth"])
            # save the condition images
            depth_denormalize = (batch["depth"] + 1) / 2.0
            depth_numpy = depth_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[0]
            depth_numpy = (depth_numpy * 255).round().astype("uint8")
            depth_pil = Image.fromarray(depth_numpy)
            depth_pil.save(os.path.join(save_path, f"depth.png"))
            batch["depth"] = batch["depth"][0].unsqueeze(0)
        if "normal" in args.cond_type:
            # structural_cond.append(batch["normal"])
            normal_denormalize = (batch["normal"] + 1) / 2.0
            normal_numpy = normal_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[0]
            normal_numpy = (normal_numpy * 255).round().astype("uint8")
            normal_pil = Image.fromarray(normal_numpy)
            normal_pil.save(os.path.join(save_path, f"normal.png"))
            batch["normal"] = batch["normal"][0].unsqueeze(0)
        if "canny" in args.cond_type:
            # structural_cond.append(batch["canny"])
            canny_denormalize = (batch["canny"] + 1) / 2.0
            canny_numpy = canny_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[0]
            canny_numpy = (canny_numpy * 255).round().astype("uint8")
            canny_pil = Image.fromarray(canny_numpy)
            canny_pil.save(os.path.join(save_path, f"canny.png"))
            batch["canny"] = batch["canny"][0].unsqueeze(0)
        if "body" in args.cond_type:
            # structural_cond.append(batch["body"])
            body_denormalize = (batch["body"] + 1) / 2.0
            body_numpy = body_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[0]
            body_numpy = (body_numpy * 255).round().astype("uint8")
            body_pil = Image.fromarray(body_numpy)
            body_pil.save(os.path.join(save_path, f"body.png"))
            batch["body"] = batch["body"][0].unsqueeze(0)
        if "face" in args.cond_type:
            # structural_cond.append(batch["face"])
            face_denormalize = (batch["face"] + 1) / 2.0
            face_numpy = face_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[0]
            face_numpy = (face_numpy * 255).round().astype("uint8")
            face_pil = Image.fromarray(face_numpy)
            face_pil.save(os.path.join(save_path, f"face.png"))
            batch["face"] = batch["face"][0].unsqueeze(0)
        if "hand" in args.cond_type:
            # structural_cond.append(batch["hand"])
            hand_denormalize = (batch["hand"] + 1) / 2.0
            hand_numpy = hand_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[0]
            hand_numpy = (hand_numpy * 255).round().astype("uint8")
            hand_pil = Image.fromarray(hand_numpy)
            hand_pil.save(os.path.join(save_path, f"hand.png"))
            batch["hand"] = batch["hand"][0].unsqueeze(0)
        
        # structural_cond = torch.cat(structural_cond, dim=1)
        
    images = []
    for i in range(len(args.validation_prompts)):
        with torch.autocast("cuda"):
            # if args.cond_inject == "spade":     
            #     image = pipeline(args.validation_prompts[i], num_inference_steps=500, generator=generator, spade=True, structural_cond=structural_cond[0].unsqueeze(0)).images[0]
            # else:
            output = pipeline(
                args.validation_prompts[i], 
                num_inference_steps=50, 
                height=args.resolution,
                width=args.resolution,
                generator=generator, 
                batch=batch, 
                args=args, 
                depth_embedder=depth_embedder, 
                normal_embedder=normal_embedder,
                canny_embedder=canny_embedder,
                body_embedder=body_embedder, 
                face_embedder=face_embedder,
                hand_embedder=hand_embedder,
                guidance_rescale=0.7 if args.flaw else 0.,
            )
            image = output.images[0]
            image.save(os.path.join(save_path, f"{i}.png"))
            if "depth" in args.noisy_cond:
                depth_image = output.depth_image[0]
                depth_image.save(os.path.join(save_path, f"{i}-depth.png"))
            if "normal" in args.noisy_cond:
                normal_image = output.normal_image[0]
                normal_image.save(os.path.join(save_path, f"{i}-normal.png"))
            if "canny" in args.noisy_cond:
                canny_image = output.canny_image[0]
                canny_image.save(os.path.join(save_path, f"{i}-canny.png"))
            if "body" in args.noisy_cond:
                body_image = output.body_image[0]
                body_image.save(os.path.join(save_path, f"{i}-body.png"))
            if "face" in args.noisy_cond:
                face_image = output.face_image[0]
                face_image.save(os.path.join(save_path, f"{i}-face.png"))
            if "hand" in args.noisy_cond:
                hand_image = output.hand_image[0]
                hand_image.save(os.path.join(save_path, f"{i}-hand.png"))
        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    ################################### newly added args ###################################
    # parser.add_argument('--prediction_type', type=str, default='v_prediction', choices=['epsilon', 'v_prediction', 'target'], help='Select a mode')
    parser.add_argument('--target_change', type=str, choices=['depth', 'normal', 'canny', 'body', 'face', 'hand'], help='how to inject the spatial condition')
    parser.add_argument('--size_cond', default=False, action="store_true")
    parser.add_argument('--flaw', default=False, action="store_true")
    parser.add_argument('--only_attn', default=False, action="store_true")
    parser.add_argument('--only_ca', default=False, action="store_true")
    parser.add_argument('--filter_mface', default=False, action="store_true")
    parser.add_argument('--filter_wpose', default=False, action="store_true")
    parser.add_argument('--filter_lowres', default=False, action="store_true")
    parser.add_argument("--filter_res", type=int)
    parser.add_argument("--validation_steps", type=int, default=500, help="Run validation every X epochs.")
    parser.add_argument('--noisy_cond', type=str, default=[], nargs="+", help='add which types of conditions')
    parser.add_argument('--dropout', default=False, action="store_true")
    parser.add_argument('--grid_dnc', default=False, action="store_true")
    parser.add_argument('--blip_concat', default=False, action="store_true")
    parser.add_argument('--string_concat', default=False, action="store_true")
    parser.add_argument('--string_substitute', default=False, action="store_true")
    parser.add_argument('--pred_cond', default=False, action="store_true")
    parser.add_argument("--depth_weight", type=float, default=0.1)
    parser.add_argument("--normal_weight", type=float, default=0.1)
    parser.add_argument("--canny_weight", type=float, default=0.1)
    parser.add_argument("--body_weight", type=float, default=0.1)
    parser.add_argument("--face_weight", type=float, default=0.1)
    parser.add_argument("--hand_weight", type=float, default=0.1)
    parser.add_argument('--cond_reshape', type=str, choices=['resize', 'vae', 'learn_conv'], help='how to reshape the spatial condition to the same shape as the latent space size')
    parser.add_argument('--cond_inject', type=str, choices=['concat', 'spade', 'sum'], help='how to inject the spatial condition')
    parser.add_argument('--cond_type', type=str, default=[], nargs="+", help='add which types of conditions')
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
    
    tokenizer = T5Tokenizer.from_pretrained(
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
        text_encoder = T5EncoderModel.from_pretrained(
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
        unet = UNet2DConditionModel.from_pretrained_2d(args.pretrained_model_name_or_path, subfolder="unet", size_cond=args.size_cond)
        unet.config.size_cond = args.size_cond
        unet.config["size_cond"] = args.size_cond
        # unet = UNet2DConditionModel.from_pretrained(
        #     args.pretrained_model_name_or_path,
        #     subfolder="unet",
        #     revision=args.non_ema_revision
        # )
        num_cond_channel = 0
        num_cond = 0
        if "depth" in args.cond_type: num_cond += 1
        if "normal" in args.cond_type: num_cond += 1
        if "canny" in args.cond_type: num_cond += 1
        if "body" in args.cond_type: num_cond += 1
        if "face" in args.cond_type: num_cond += 1
        if "hand" in args.cond_type: num_cond += 1
        
        if args.cond_inject == "concat":
            num_per_cond_channel = 3 if args.cond_reshape == "resize" else 4
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
            unet.conv_in.weight[:, :4] = weights
            # unet.conv_in.weight[:, :4] = weights * 1. / (num_cond + 1) # original weights
            unet.conv_in.weight[:, 4:] = torch.zeros(unet.conv_in.weight[:, 4:].shape) # new weights initialized to zero
            # unet.conv_in.bias[:4] = torch.nn.Parameter(bias * 1. / (num_cond + 1))
            unet.conv_in.bias = torch.nn.Parameter(bias)
            # for i in range(num_cond):
            #     unet.conv_in.weight[:, 4 * (i + 1) : 4 * (i + 2)] = weights * 1. / (num_cond + 1)
            #     # unet.conv_in.bias[4 * (i + 1) : 4 * (i + 2)] = torch.nn.Parameter(bias * 1. / (num_cond + 1))
                
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
                unet.conv_out.bias[:4] = torch.nn.Parameter(bias)
                for i in range(num_cond):
                    unet.conv_out.weight[4 * (i + 1) : 4 * (i + 2)] = weights
                    unet.conv_out.bias[4 * (i + 1) : 4 * (i + 2)] = torch.nn.Parameter(bias)
                # unet.conv_out.weight[4:] = torch.zeros(unet.conv_out.weight[4:].shape) # new weights initialized to zero
                # unet.conv_out.bias[4:] = torch.zeros(unet.conv_out.bias[4:].shape)

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
            ema_unet = UNet2DConditionModel.from_pretrained_2d(args.pretrained_model_name_or_path, subfolder="unet", size_cond=args.size_cond)
            ema_unet.config.size_cond = args.size_cond
            ema_unet.config["size_cond"] = args.size_cond
            # ema_unet = UNet2DConditionModel.from_pretrained(
            #     args.pretrained_model_name_or_path,
            #     subfolder="unet",
            #     revision=args.non_ema_revision
            # )
            num_cond_channel = 0
            num_cond = 0
            if "depth" in args.cond_type: num_cond += 1
            if "normal" in args.cond_type: num_cond += 1
            if "canny" in args.cond_type: num_cond += 1
            if "body" in args.cond_type: num_cond += 1
            if "face" in args.cond_type: num_cond += 1
            if "hand" in args.cond_type: num_cond += 1
                
            if args.cond_inject == "concat":
                num_per_cond_channel = 3 if args.cond_reshape == "resize" else 4
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
                ema_unet.conv_in.weight[:, :4] = weights
                # ema_unet.conv_in.weight[:, :4] = weights * 1. / (num_cond + 1) # original weights
                ema_unet.conv_in.weight[:, 4:] = torch.zeros(ema_unet.conv_in.weight[:, 4:].shape) # new weights initialized to zero
                # ema_unet.conv_in.bias[:4] = torch.nn.Parameter(bias * 1. / (num_cond + 1))
                ema_unet.conv_in.bias = torch.nn.Parameter(bias)
                # for i in range(num_cond):
                #     ema_unet.conv_in.weight[:, 4 * (i + 1) : 4 * (i + 2)] = weights * 1. / (num_cond + 1)
                #     # ema_unet.conv_in.bias[4 * (i + 1) : 4 * (i + 2)] = torch.nn.Parameter(bias * 1. / (num_cond + 1))
                
            if args.pred_cond:
                num_cond = 0
                if "depth" in args.cond_type: num_cond += 1
                if "normal" in args.cond_type: num_cond += 1
                if "canny" in args.cond_type: num_cond += 1
                if "body" in args.cond_type: num_cond += 1
                if "face" in args.cond_type: num_cond += 1
                if "hand" in args.cond_type: num_cond += 1
                ema_unet.config.out_channels = 4 + num_cond * 4
                ema_unet.config["out_channels"] = 4 + num_cond * 4

                # Modify input layer to have additional structural condition channels
                weights = ema_unet.conv_out.weight.clone()
                bias = ema_unet.conv_out.bias.clone()

                ema_unet.conv_out = torch.nn.Conv2d(weights.shape[1], 4 + num_cond * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                with torch.no_grad():
                    ema_unet.conv_out.weight[:4] = weights # original weights
                    ema_unet.conv_out.bias[:4] = torch.nn.Parameter(bias)
                    for i in range(num_cond):
                        ema_unet.conv_out.weight[4 * (i + 1) : 4 * (i + 2)] = weights
                        ema_unet.conv_out.bias[4 * (i + 1) : 4 * (i + 2)] = torch.nn.Parameter(bias)
                    # ema_unet.conv_out.weight[4:] = torch.zeros(ema_unet.conv_out.weight[4:].shape) # new weights initialized to zero
                    # ema_unet.conv_out.bias[4:] = torch.zeros(ema_unet.conv_out.bias[4:].shape)
                
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
    data_args.train_data = []
    for data_folder in args.train_data_dir:
        data_list = [os.path.join(data_folder, x)
                                 for x in os.listdir(data_folder) if
                                 x.endswith('.tar')]  # open('s3_urls.txt').read().splitlines()
        data_list.sort()
        if "getty" in data_folder:
            data_list = data_list[50:]
        data_args.train_data += data_list

    random.shuffle(data_args.train_data)

    print(f'Found {len(data_args.train_data)} .tar files in {args.train_data_dir}')
    data_args.train_num_samples = 400000000
    data_args.train_data_upsampling_factors = None
    data_args.batch_size = args.train_batch_size
    data_args.world_size = torch.distributed.get_world_size()
    # print(torch.distributed.get_world_size())
    data_args.workers = args.dataloader_num_workers
    data_args.seed = -1
    train_dataset = get_wds_dataset_cond(data_args,
                                  main_args=args,
                                  is_train=True,
                                  epoch=0,
                                  floor=False,
                                  tokenizer=tokenizer,
                                  dropout=args.dropout,
                                  string_concat=args.string_concat,
                                  string_substitute=args.string_substitute,
                                  grid_dnc=args.grid_dnc,
                                  filter_lowres=args.filter_lowres,
                                  filter_res=args.filter_res,
                                  filter_mface=args.filter_mface,
                                  filter_wpose=args.filter_wpose,
                                )
    # train_dataset = train_dataset.with_length(300000000)
    train_dataloader = train_dataset.dataloader

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
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
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

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        tracker_config.pop("train_data_dir")
        tracker_config.pop("cond_type")
        tracker_config.pop("noisy_cond")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num examples = {400000000}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

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

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    # save the pretrained SD generation results
    if accelerator.is_main_process:
        # if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
        if args.validation_prompts is not None:
            log_validation(
                vae,
                text_encoder,
                tokenizer,
                unet,
                args,
                accelerator,
                weight_dtype,
                global_step,
            )
            
    def compute_embeddings(ori_h, ori_w, crops_coords_top_left_h, crops_coords_top_left_w):
        original_size = (ori_h, ori_w)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (crops_coords_top_left_h, crops_coords_top_left_w)

        with torch.no_grad():

            # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])
            add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
            unet_added_cond_kwargs = {"time_ids": add_time_ids}

        return unet_added_cond_kwargs

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.cond_reshape == "learn_conv":
            if "depth" in args.cond_type:
                depth_embedder.train()
            if "normal" in args.cond_type:
                normal_embedder.train()
            if "canny" in args.cond_type:
                canny_embedder.train()
            if "body" in args.cond_type:
                body_embedder.train()
            if "face" in args.cond_type:
                face_embedder.train()
            if "hand" in args.cond_type:
                hand_embedder.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % args.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue
            if args.resume_from_checkpoint:
                step += resume_step
            
            # try:
            # images, text_input_ids = batch
            images, text_input_ids, text_raw, blip, blip_raw, body, face, hand, normal, depth, canny, descriptions = batch
            if args.target_change is not None:
                if args.target_change == 'depth':
                    images = depth
                if args.target_change == 'normal':
                    images = normal
                if args.target_change == 'canny':
                    images = canny
                if args.target_change == 'body':
                    images = body
                if args.target_change == 'face':
                    images = face
                if args.target_change == 'hand':
                    images = hand
            # images, text_input_ids, blip, normal, depth, canny = batch
            images = images.to(unet.device)
            text_input_ids = text_input_ids.to(text_encoder.device)
            text_input_ids = text_input_ids.squeeze(dim=1)
            add_time_ids = []
            for i_sample, description in enumerate(descriptions):
                instance_unet_added_conditions = compute_embeddings(
                    description["h"], description["w"], 
                    description["crop_tl_h"], description["crop_tl_w"],
                )
                add_time_ids.append(instance_unet_added_conditions["time_ids"])
            add_time_ids = torch.cat(add_time_ids, dim=0).to(text_encoder.device)
            batch = {
                'pixel_values': images,
                'input_ids': text_input_ids,
                "unet_added_conditions": {"time_ids": add_time_ids},
            }
            
            if "blip" in args.cond_type:
                blip = blip.to(text_encoder.device)
                blip = blip.squeeze(dim=1)
                batch["blip"] = blip
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

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                _, _, h, w = latents.shape
                
                if "depth" in args.cond_type:
                    if args.cond_reshape == "vae":
                        depth_latents = vae.encode(batch["depth"].to(weight_dtype)).latent_dist.sample()
                        depth_latents = depth_latents * vae.config.scaling_factor
                    elif args.cond_reshape == "resize":
                        depth_latents = F.interpolate(batch['depth'], (h,w))
                    elif args.cond_reshape == "learn_conv":
                        depth_latents = depth_embedder(batch['depth'])
                    else:
                        assert False, "unknown condition reshape type"
                        
                if "normal" in args.cond_type:
                    if args.cond_reshape == "vae":
                        normal_latents = vae.encode(batch["normal"].to(weight_dtype)).latent_dist.sample()
                        normal_latents = normal_latents * vae.config.scaling_factor
                    elif args.cond_reshape == "resize":
                        normal_latents = F.interpolate(batch['normal'], (h,w))
                    elif args.cond_reshape == "learn_conv":
                        normal_latents = normal_embedder(batch['normal'])
                    else:
                        assert False, "unknown condition reshape type"
                        
                if "canny" in args.cond_type:
                    if args.cond_reshape == "vae":
                        canny_latents = vae.encode(batch["canny"].to(weight_dtype)).latent_dist.sample()
                        canny_latents = canny_latents * vae.config.scaling_factor
                    elif args.cond_reshape == "resize":
                        canny_latents = F.interpolate(batch['canny'], (h,w))
                    elif args.cond_reshape == "learn_conv":
                        canny_latents = canny_embedder(batch['canny'])
                    else:
                        assert False, "unknown condition reshape type"
                        
                if "body" in args.cond_type:
                    if args.cond_reshape == "vae":
                        body_latents = vae.encode(batch["body"].to(weight_dtype)).latent_dist.sample()
                        body_latents = body_latents * vae.config.scaling_factor
                    elif args.cond_reshape == "resize":
                        body_latents = F.interpolate(batch['body'], (h,w))
                    elif args.cond_reshape == "learn_conv":
                        body_latents = body_embedder(batch['body'])
                    else:
                        assert False, "unknown condition reshape type"
                        
                if "face" in args.cond_type:
                    if args.cond_reshape == "vae":
                        face_latents = vae.encode(batch["face"].to(weight_dtype)).latent_dist.sample()
                        face_latents = face_latents * vae.config.scaling_factor
                    elif args.cond_reshape == "resize":
                        face_latents = F.interpolate(batch['face'], (h,w))
                    elif args.cond_reshape == "learn_conv":
                        face_latents = face_embedder(batch['face'])
                    else:
                        assert False, "unknown condition reshape type"
                        
                if "hand" in args.cond_type:
                    if args.cond_reshape == "vae":
                        hand_latents = vae.encode(batch["hand"].to(weight_dtype)).latent_dist.sample()
                        hand_latents = hand_latents * vae.config.scaling_factor
                    elif args.cond_reshape == "resize":
                        hand_latents = F.interpolate(batch['hand'], (h,w))
                    elif args.cond_reshape == "learn_conv":
                        hand_latents = hand_embedder(batch['hand'])
                    else:
                        assert False, "unknown condition reshape type"
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    
                if "depth" in args.cond_type and "depth" in args.noisy_cond:
                    noise_depth = torch.randn_like(depth_latents)
                    ori_depth_latents = depth_latents
                    depth_latents = noise_scheduler.add_noise(depth_latents, noise_depth, timesteps)
                    
                if "normal" in args.cond_type and "normal" in args.noisy_cond:
                    noise_normal = torch.randn_like(normal_latents)
                    ori_normal_latents = normal_latents
                    normal_latents = noise_scheduler.add_noise(normal_latents, noise_normal, timesteps)
                    
                if "canny" in args.cond_type and "canny" in args.noisy_cond:
                    noise_canny = torch.randn_like(canny_latents)
                    ori_canny_latents = canny_latents
                    canny_latents = noise_scheduler.add_noise(canny_latents, noise_canny, timesteps)
                    
                if "body" in args.cond_type and "body" in args.noisy_cond:
                    noise_body = torch.randn_like(body_latents)
                    ori_body_latents = body_latents
                    body_latents = noise_scheduler.add_noise(body_latents, noise_body, timesteps)
                    
                if "face" in args.cond_type and "face" in args.noisy_cond:
                    noise_face = torch.randn_like(face_latents)
                    ori_face_latents = face_latents
                    face_latents = noise_scheduler.add_noise(face_latents, noise_face, timesteps)
                    
                if "hand" in args.cond_type and "hand" in args.noisy_cond:
                    noise_hand = torch.randn_like(hand_latents)
                    ori_hand_latents = hand_latents
                    hand_latents = noise_scheduler.add_noise(hand_latents, noise_hand, timesteps)    
                    
                if args.cond_inject == "concat":
                    noisy_latents = torch.cat([noisy_latents, depth_latents], dim=1) if "depth" in args.cond_type else noisy_latents
                    noisy_latents = torch.cat([noisy_latents, normal_latents], dim=1) if "normal" in args.cond_type else noisy_latents
                    noisy_latents = torch.cat([noisy_latents, canny_latents], dim=1) if "canny" in args.cond_type else noisy_latents
                    noisy_latents = torch.cat([noisy_latents, body_latents], dim=1) if "body" in args.cond_type else noisy_latents
                    noisy_latents = torch.cat([noisy_latents, face_latents], dim=1) if "face" in args.cond_type else noisy_latents
                    noisy_latents = torch.cat([noisy_latents, hand_latents], dim=1) if "hand" in args.cond_type else noisy_latents
                elif args.cond_inject == "sum":
                    if args.cond_reshape == "vae":
                        channel_dim = 4
                    elif args.cond_reshape == "resize":
                        channel_dim = 3
                    elif args.cond_reshape == "learn_conv":
                        channel_dim = args.embedder_channel
                    sum_latents = torch.zeros((noisy_latents.shape[0], channel_dim, h, w)).to(unet.device)
                    sum_latents = sum_latents + depth_latents if "depth" in args.cond_type else sum_latents
                    sum_latents = sum_latents + normal_latents if "normal" in args.cond_type else sum_latents
                    sum_latents = sum_latents + canny_latents if "canny" in args.cond_type else sum_latents
                    sum_latents = sum_latents + body_latents if "body" in args.cond_type else sum_latents
                    sum_latents = sum_latents + face_latents if "face" in args.cond_type else sum_latents
                    sum_latents = sum_latents + hand_latents if "hand" in args.cond_type else sum_latents
                    noisy_latents = torch.cat([noisy_latents, sum_latents], dim=1)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                if "blip" in args.cond_type and args.blip_concat:
                    blip_hidden_states = text_encoder(batch["blip"])[0]
                    encoder_hidden_states = torch.cat([encoder_hidden_states, blip_hidden_states], dim=1)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                    noise_scheduler.register_to_config(rescale_betas_zero_snr=args.flaw)
                    noise_scheduler.register_to_config(timestep_spacing="trailing" if args.flaw else "leading")

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                    if "depth" in args.cond_type and "depth" in args.noisy_cond:
                        target_depth = noise_depth
                    if "normal" in args.cond_type and "normal" in args.noisy_cond:
                        target_normal = noise_normal
                    if "canny" in args.cond_type and "canny" in args.noisy_cond:
                        target_canny = noise_canny
                    if "body" in args.cond_type and "body" in args.noisy_cond:
                        target_body = noise_body
                    if "face" in args.cond_type and "face" in args.noisy_cond:
                        target_face = noise_face
                    if "hand" in args.cond_type and "hand" in args.noisy_cond:
                        target_hand = noise_hand
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    if "depth" in args.cond_type and "depth" in args.noisy_cond:
                        target_depth = noise_scheduler.get_velocity(ori_depth_latents, noise, timesteps)
                    if "normal" in args.cond_type and "normal" in args.noisy_cond:
                        target_normal = noise_scheduler.get_velocity(ori_normal_latents, noise, timesteps)
                    if "canny" in args.cond_type and "canny" in args.noisy_cond:
                        target_canny = noise_scheduler.get_velocity(ori_canny_latents, noise, timesteps)
                    if "body" in args.cond_type and "body" in args.noisy_cond:
                        target_body = noise_scheduler.get_velocity(ori_body_latents, noise, timesteps)
                    if "face" in args.cond_type and "face" in args.noisy_cond:
                        target_face = noise_scheduler.get_velocity(ori_face_latents, noise, timesteps)
                    if "hand" in args.cond_type and "hand" in args.noisy_cond:
                        target_hand = noise_scheduler.get_velocity(ori_hand_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                if args.cond_inject == "spade":
                    structural_cond = []
                    if "depth" in args.cond_type:
                        structural_cond.append(batch["depth"])
                    if "normal" in args.cond_type:
                        structural_cond.append(batch["normal"])
                    if "canny" in args.cond_type:
                        structural_cond.append(batch["canny"])
                    if "body" in args.cond_type:
                        structural_cond.append(batch["body"])
                    if "face" in args.cond_type:
                        structural_cond.append(batch["face"])
                    if "hand" in args.cond_type:
                        structural_cond.append(batch["hand"])
                    structural_cond = torch.cat(structural_cond, dim=1)
                    model_pred = unet(noisy_latents, structural_cond, timesteps, encoder_hidden_states, added_cond_kwargs=batch["unet_added_conditions"]).sample
                else:
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=batch["unet_added_conditions"]).sample
                    
                if model_pred.shape[1] > 4:
                    cond_pred = model_pred[:, 4:]
                    model_pred = model_pred[:, :4]
                    if "depth" in args.cond_type and "depth" in args.noisy_cond:
                        depth_pred = cond_pred[:, :4]
                        if cond_pred.shape[1] > 4:
                            cond_pred = cond_pred[:, 4:]
                    if "normal" in args.cond_type and "normal" in args.noisy_cond:
                        normal_pred = cond_pred[:, :4]
                        if cond_pred.shape[1] > 4:
                            cond_pred = cond_pred[:, 4:]
                    if "canny" in args.cond_type and "canny" in args.noisy_cond:
                        canny_pred = cond_pred[:, :4]
                        if cond_pred.shape[1] > 4:
                            cond_pred = cond_pred[:, 4:]
                    if "body" in args.cond_type and "body" in args.noisy_cond:
                        body_pred = cond_pred[:, :4]
                        if cond_pred.shape[1] > 4:
                            cond_pred = cond_pred[:, 4:]
                    if "face" in args.cond_type and "face" in args.noisy_cond:
                        face_pred = cond_pred[:, :4]
                        if cond_pred.shape[1] > 4:
                            cond_pred = cond_pred[:, 4:]
                    if "hand" in args.cond_type and "hand" in args.noisy_cond:
                        hand_pred = cond_pred[:, :4]
                        if cond_pred.shape[1] > 4:
                            cond_pred = cond_pred[:, 4:]   
                
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                    
                dm_loss = loss    
                    
                if "depth" in args.cond_type and "depth" in args.noisy_cond:
                    depth_loss = F.mse_loss(depth_pred.float(), target_depth.float(), reduction="mean") * args.depth_weight
                    loss += depth_loss
                if "normal" in args.cond_type and "normal" in args.noisy_cond:
                    normal_loss = F.mse_loss(normal_pred.float(), target_normal.float(), reduction="mean") * args.normal_weight
                    loss += normal_loss
                if "canny" in args.cond_type and "canny" in args.noisy_cond:
                    canny_loss = F.mse_loss(canny_pred.float(), target_canny.float(), reduction="mean") * args.canny_weight
                    loss += canny_loss
                if "body" in args.cond_type and "body" in args.noisy_cond:
                    body_loss = F.mse_loss(body_pred.float(), target_body.float(), reduction="mean") * args.body_weight
                    loss += body_loss
                if "face" in args.cond_type and "face" in args.noisy_cond:
                    face_loss = F.mse_loss(face_pred.float(), target_face.float(), reduction="mean") * args.face_weight
                    loss += face_loss
                if "hand" in args.cond_type and "hand" in args.noisy_cond:
                    hand_loss = F.mse_loss(hand_pred.float(), target_hand.float(), reduction="mean") * args.hand_weight
                    loss += hand_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        if args.cond_reshape == "learn_conv":
                            if "depth" in args.cond_type:
                                torch.save(depth_embedder.state_dict(), os.path.join(save_path, "depth_embedder.pth"))
                            if "normal" in args.cond_type:
                                torch.save(normal_embedder.state_dict(), os.path.join(save_path, "normal_embedder.pth"))
                            if "canny" in args.cond_type:
                                torch.save(canny_embedder.state_dict(), os.path.join(save_path, "canny_embedder.pth"))
                            if "body" in args.cond_type:
                                torch.save(body_embedder.state_dict(), os.path.join(save_path, "body_embedder.pth"))
                            if "face" in args.cond_type:
                                torch.save(face_embedder.state_dict(), os.path.join(save_path, "face_embedder.pth"))
                            if "hand" in args.cond_type:
                                torch.save(hand_embedder.state_dict(), os.path.join(save_path, "hand_embedder.pth"))

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "dm_loss": dm_loss.detach().item()}
            if "depth" in args.cond_type and "depth" in args.noisy_cond:
                logs["depth"] = depth_loss.detach().item()
            if "normal" in args.cond_type and "normal" in args.noisy_cond:
                logs["normal"] = normal_loss.detach().item()
            if "canny" in args.cond_type and "canny" in args.noisy_cond:
                logs["canny"] = canny_loss.detach().item()
            if "body" in args.cond_type and "body" in args.noisy_cond:
                logs["body"] = body_loss.detach().item()
            if "face" in args.cond_type and "face" in args.noisy_cond:
                logs["face"] = face_loss.detach().item()
            if "hand" in args.cond_type and "hand" in args.noisy_cond:
                logs["hand"] = hand_loss.detach().item()
            progress_bar.set_postfix(**logs)

            if accelerator.is_main_process:
                # if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                        
                    log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        unet,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                        batch,
                        depth_embedder,
                        normal_embedder,
                        canny_embedder,
                        body_embedder,
                        face_embedder,
                        hand_embedder,
                    )
                        
                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())

            if global_step >= args.max_train_steps:
                break
                
            # except Exception as e:
            #     print(f"An error occurred: {str(e)}")
                
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
