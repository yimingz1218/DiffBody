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
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    # ControlNetModel,
    DDPMScheduler,
    # StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.models.controlnet_composer import ControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from openclip.training.data import get_wds_dataset, get_wds_dataset_cond
from pipelines.pipeline_controlnet_composer import StableDiffusionControlNetPipeline
import boto3
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.19.0.dev0")

logger = get_logger(__name__)


def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step, test_dataloader):
    save_path = os.path.join(args.output_dir, "images", f"image-{step}")
    if os.path.exists(save_path):
        return
    os.makedirs(save_path, exist_ok=True)
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
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

    for i_test, batch in enumerate(test_dataloader):
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
        if "midas_depth" in args.cond_type:
            midas_depth = midas_depth.to(unet.device)
            batch["midas_depth"] = midas_depth
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
        if "whole" in args.cond_type:
            whole = whole.to(unet.device)
            batch["whole"] = whole
            
        for i_batch in range(batch_size):
            img_id = i_test * batch_size + i_batch
            image_denormalize = (batch["pixel_values"] + 1) / 2.0
            image_numpy = image_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
            image_numpy = (image_numpy * 255).round().astype("uint8")
            image_pil = Image.fromarray(image_numpy)
            image_pil.save(os.path.join(save_path, f"image-{img_id}.png"))
            if "depth" in args.cond_type:
                depth_denormalize = (batch["depth"] + 1) / 2.0
                depth_numpy = depth_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                depth_numpy = (depth_numpy * 255).round().astype("uint8")
                depth_pil = Image.fromarray(depth_numpy)
                depth_pil.save(os.path.join(save_path, f"depth-{img_id}.png"))
                batch["depth"] = batch["depth"][0].unsqueeze(0)
            if "midas_depth" in args.cond_type:
                midas_depth_denormalize = (batch["midas_depth"] + 1) / 2.0
                midas_depth_numpy = midas_depth_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                midas_depth_numpy = (midas_depth_numpy * 255).round().astype("uint8")
                midas_depth_pil = Image.fromarray(midas_depth_numpy)
                midas_depth_pil.save(os.path.join(save_path, f"midas_depth-{img_id}.png"))
                batch["midas_depth"] = batch["midas_depth"][0].unsqueeze(0)
            if "normal" in args.cond_type:
                normal_denormalize = (batch["normal"] + 1) / 2.0
                normal_numpy = normal_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                normal_numpy = (normal_numpy * 255).round().astype("uint8")
                normal_pil = Image.fromarray(normal_numpy)
                normal_pil.save(os.path.join(save_path, f"normal-{img_id}.png"))
                batch["normal"] = batch["normal"][0].unsqueeze(0)
            if "canny" in args.cond_type:
                canny_denormalize = (batch["canny"] + 1) / 2.0
                canny_numpy = canny_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                canny_numpy = (canny_numpy * 255).round().astype("uint8")
                canny_pil = Image.fromarray(canny_numpy)
                canny_pil.save(os.path.join(save_path, f"canny-{img_id}.png"))
                batch["canny"] = batch["canny"][0].unsqueeze(0)
            if "body" in args.cond_type:
                body_denormalize = (batch["body"] + 1) / 2.0
                body_numpy = body_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                body_numpy = (body_numpy * 255).round().astype("uint8")
                body_pil = Image.fromarray(body_numpy)
                body_pil.save(os.path.join(save_path, f"body-{img_id}.png"))
                batch["body"] = batch["body"][0].unsqueeze(0)
            if "face" in args.cond_type:
                face_denormalize = (batch["face"] + 1) / 2.0
                face_numpy = face_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                face_numpy = (face_numpy * 255).round().astype("uint8")
                face_pil = Image.fromarray(face_numpy)
                face_pil.save(os.path.join(save_path, f"face-{img_id}.png"))
                batch["face"] = batch["face"][0].unsqueeze(0)
            if "hand" in args.cond_type:
                hand_denormalize = (batch["hand"] + 1) / 2.0
                hand_numpy = hand_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                hand_numpy = (hand_numpy * 255).round().astype("uint8")
                hand_pil = Image.fromarray(hand_numpy)
                hand_pil.save(os.path.join(save_path, f"hand-{img_id}.png"))
                batch["hand"] = batch["hand"][0].unsqueeze(0)
            if "whole" in args.cond_type:
                whole_denormalize = (batch["whole"] + 1) / 2.0
                whole_numpy = whole_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                whole_numpy = (whole_numpy * 255).round().astype("uint8")
                whole_pil = Image.fromarray(whole_numpy)
                whole_pil.save(os.path.join(save_path, f"whole-{img_id}.png"))
                batch["whole"] = batch["whole"][0].unsqueeze(0)
                
            controlnet_image = []
            for key in ['depth', 'midas_depth', 'normal', 'canny', 'body', 'face', 'hand', 'whole']:
                if key in args.cond_type:
                    controlnet_image.append(batch[key][i_batch])
                
            with torch.autocast("cuda"):
                output = pipeline(
                    batch["text_raw"][i_batch], 
                    image=controlnet_image, 
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=50, 
                    generator=generator,
                    negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                )

                image = output.images[0]
                image.save(os.path.join(save_path, f"{img_id}.png"))
                
        if i_test >= 15:
            break
    return

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument('--inference_folder_name', type=str)
    parser.add_argument('--overfit', default=False, action="store_true")
    parser.add_argument('--prediction_type', type=str, default='epsilon', choices=['epsilon', 'v_prediction', 'sample'], help='Select a mode')
    parser.add_argument("--center_crop",default=False,action="store_true",)
    parser.add_argument("--random_flip",default=False,action="store_true",)
    parser.add_argument("--timestep_start", type=int, default=0)
    parser.add_argument("--timestep_end", type=int, default=1000)
    parser.add_argument("--test_data_dir", nargs='+', type=str, default=None)
    parser.add_argument("--cond_num", type=int, default=3)
    parser.add_argument("--distill_weight", type=float, default=1.0)
    parser.add_argument('--fusion', type=str, default="sum")
    parser.add_argument('--reserve_minus_one_to_one', default=True, action="store_false")
    parser.add_argument('--hierarchy_distill', default=False, action="store_true")
    parser.add_argument("--train_data_dir",nargs='+',type=str,default=None)
    parser.add_argument('--cond_type', type=str, default=[], nargs="+", help='add which types of conditions')
    parser.add_argument('--cond_type_test', type=str, default=[], nargs="+", help='add which types of conditions')
    parser.add_argument('--dropout', default=False, action="store_true")
    parser.add_argument('--grid_dnc', default=False, action="store_true")
    parser.add_argument('--blip_concat', default=False, action="store_true")
    parser.add_argument('--string_concat', default=False, action="store_true")
    parser.add_argument('--string_substitute', default=False, action="store_true")
    parser.add_argument('--filter_mface', default=False, action="store_true")
    parser.add_argument('--filter_wpose', default=False, action="store_true")
    parser.add_argument('--filter_lowres', default=False, action="store_true")
    parser.add_argument("--filter_res", type=int)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
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
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
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
        default=5e-6,
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
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
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
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
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
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
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
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    # if args.validation_prompt is not None and args.validation_image is None:
    #     raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    # if args.validation_prompt is None and args.validation_image is not None:
    #     raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    # if (
    #     args.validation_image is not None
    #     and args.validation_prompt is not None
    #     and len(args.validation_image) != 1
    #     and len(args.validation_prompt) != 1
    #     and len(args.validation_image) != len(args.validation_prompt)
    # ):
    #     raise ValueError(
    #         "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
    #         " or the same number of `--validation_prompt`s and `--validation_image`s"
    #     )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=18000))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
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
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
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

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.config.prediction_type = args.prediction_type
    noise_scheduler.config['prediction_type'] = args.prediction_type
    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    vae = AutoencoderKL.from_pretrained("/fsx_laion/alvin/pretrain/sd-vae-ft-mse", subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet, cond_num=args.cond_num, fusion=args.fusion, normalize_to_0_1=args.reserve_minus_one_to_one)
    controlnet.config.cond_num = args.cond_num
    controlnet.config["cond_num"] = args.cond_num
    controlnet.config.fusion = args.fusion
    controlnet.config["fusion"] = args.fusion
    controlnet.config.normalize_to_0_1 = args.reserve_minus_one_to_one
    controlnet.config["normalize_to_0_1"] = args.reserve_minus_one_to_one

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
    text_encoder.requires_grad_(False)
    controlnet.eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
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

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    class Args:
        pass  # to use open clip api
    data_args = Args()
    data_args.val_data = []
    try:
        from urlparse import urlparse
    except ImportError:
        from urllib.parse import urlparse

    class S3Url(object):

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
            if args.overfit:
                data_list = [data_list[0]] * 50
            else:
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
                                  filter_res=args.filter_res,
                                  filter_mface=args.filter_mface,
                                  filter_wpose=args.filter_wpose,
                                  )
    # train_dataset = train_dataset.with_length(300000000)
    test_dataloader = test_dataset.dataloader

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
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # # Prepare everything with our `accelerator`.
    controlnet, optimizer, test_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, test_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

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
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # progress_bar = tqdm(
    #     range(0, args.max_train_steps),
    #     initial=initial_global_step,
    #     desc="Steps",
    #     # Only show the progress bar once on each machine.
    #     disable=not accelerator.is_local_main_process,
    # )
    # progress_bar.set_description("Steps")
    
    # # save the pretrained SD generation results
    # if accelerator.is_main_process and args.timestep_start == 0 and args.timestep_end == 1000:
    #     # if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
    #     log_validation(
    #         vae,
    #         text_encoder,
    #         tokenizer,
    #         unet,
    #         controlnet,
    #         args,
    #         accelerator,
    #         weight_dtype,
    #         global_step,
    #         test_dataloader
    #     )
    
    # def compute_embeddings(ori_h, ori_w, crops_coords_top_left_h, crops_coords_top_left_w):
    #     original_size = (ori_h, ori_w)
    #     target_size = (args.resolution, args.resolution)
    #     crops_coords_top_left = (crops_coords_top_left_h, crops_coords_top_left_w)

    #     with torch.no_grad():

    #         # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    #         add_time_ids = list(original_size + crops_coords_top_left + target_size)
    #         add_time_ids = torch.tensor([add_time_ids])
    #         add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
    #         unet_added_cond_kwargs = {"time_ids": add_time_ids}

    #     return unet_added_cond_kwargs
    # print((controlnet.controlnet_cond_embedding.conv_out_list[0].weight == 0).all())
    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.config.prediction_type = args.prediction_type
    pipeline.scheduler.config['prediction_type'] = args.prediction_type
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=False)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        
    save_path = os.path.join(args.output_dir, args.inference_folder_name, f"image-{global_step}")
    os.makedirs(save_path, exist_ok=True)

    for i_test, batch in enumerate(test_dataloader):
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
        if "midas_depth" in args.cond_type:
            midas_depth = midas_depth.to(unet.device)
            batch["midas_depth"] = midas_depth
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
        if "whole" in args.cond_type:
            whole = whole.to(unet.device)
            batch["whole"] = whole
            
        for i_batch in range(batch_size):
            img_id = i_test * batch_size + i_batch
            image_denormalize = (batch["pixel_values"] + 1) / 2.0
            image_numpy = image_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
            image_numpy = (image_numpy * 255).round().astype("uint8")
            image_pil = Image.fromarray(image_numpy)
            image_pil.save(os.path.join(save_path, f"image-{img_id}.png"))
            if "depth" in args.cond_type:
                depth_denormalize = (batch["depth"] + 1) / 2.0
                depth_numpy = depth_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                depth_numpy = (depth_numpy * 255).round().astype("uint8")
                depth_pil = Image.fromarray(depth_numpy)
                depth_pil.save(os.path.join(save_path, f"depth-{img_id}.png"))
                if "depth" in args.cond_type_test:
                    batch["depth"] = batch["depth"][0].unsqueeze(0)
                else:
                    batch["depth"] = (torch.ones_like(batch["depth"][0].unsqueeze(0)) * (-1)).to(unet.device)
            if "midas_depth" in args.cond_type:
                midas_depth_denormalize = (batch["midas_depth"] + 1) / 2.0
                midas_depth_numpy = midas_depth_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                midas_depth_numpy = (midas_depth_numpy * 255).round().astype("uint8")
                midas_depth_pil = Image.fromarray(midas_depth_numpy)
                midas_depth_pil.save(os.path.join(save_path, f"midas_depth-{img_id}.png"))
                if "midas_depth" in args.cond_type_test:
                    batch["midas_depth"] = batch["midas_depth"][0].unsqueeze(0)
                else:
                    batch["midas_depth"] = (torch.ones_like(batch["midas_depth"][0].unsqueeze(0)) * (-1)).to(unet.device)
            if "normal" in args.cond_type:
                normal_denormalize = (batch["normal"] + 1) / 2.0
                normal_numpy = normal_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                normal_numpy = (normal_numpy * 255).round().astype("uint8")
                normal_pil = Image.fromarray(normal_numpy)
                normal_pil.save(os.path.join(save_path, f"normal-{img_id}.png"))
                if "normal" in args.cond_type_test:
                    batch["normal"] = batch["normal"][0].unsqueeze(0)
                else:
                    batch["normal"] = (torch.ones_like(batch["normal"][0].unsqueeze(0)) * (-1)).to(unet.device)
            if "canny" in args.cond_type:
                canny_denormalize = (batch["canny"] + 1) / 2.0
                canny_numpy = canny_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                canny_numpy = (canny_numpy * 255).round().astype("uint8")
                canny_pil = Image.fromarray(canny_numpy)
                canny_pil.save(os.path.join(save_path, f"canny-{img_id}.png"))
                if "canny" in args.cond_type_test:
                    batch["canny"] = batch["canny"][0].unsqueeze(0)
                else:
                    batch["canny"] = (torch.ones_like(batch["canny"][0].unsqueeze(0)) * (-1)).to(unet.device)
            if "body" in args.cond_type:
                body_denormalize = (batch["body"] + 1) / 2.0
                body_numpy = body_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                body_numpy = (body_numpy * 255).round().astype("uint8")
                body_pil = Image.fromarray(body_numpy)
                body_pil.save(os.path.join(save_path, f"body-{img_id}.png"))
                if "body" in args.cond_type_test:
                    batch["body"] = batch["body"][0].unsqueeze(0)
                else:
                    batch["body"] = (torch.ones_like(batch["body"][0].unsqueeze(0)) * (-1)).to(unet.device)
            if "face" in args.cond_type:
                face_denormalize = (batch["face"] + 1) / 2.0
                face_numpy = face_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                face_numpy = (face_numpy * 255).round().astype("uint8")
                face_pil = Image.fromarray(face_numpy)
                face_pil.save(os.path.join(save_path, f"face-{img_id}.png"))
                if "face" in args.cond_type_test:
                    batch["face"] = batch["face"][0].unsqueeze(0)
                else:
                    batch["face"] = (torch.ones_like(batch["face"][0].unsqueeze(0)) * (-1)).to(unet.device)
            if "hand" in args.cond_type:
                hand_denormalize = (batch["hand"] + 1) / 2.0
                hand_numpy = hand_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                hand_numpy = (hand_numpy * 255).round().astype("uint8")
                hand_pil = Image.fromarray(hand_numpy)
                hand_pil.save(os.path.join(save_path, f"hand-{img_id}.png"))
                if "hand" in args.cond_type_test:
                    batch["hand"] = batch["hand"][0].unsqueeze(0)
                else:
                    batch["hand"] = (torch.ones_like(batch["hand"][0].unsqueeze(0)) * (-1)).to(unet.device)
            if "whole" in args.cond_type:
                whole_denormalize = (batch["whole"] + 1) / 2.0
                whole_numpy = whole_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[i_batch]
                whole_numpy = (whole_numpy * 255).round().astype("uint8")
                whole_pil = Image.fromarray(whole_numpy)
                whole_pil.save(os.path.join(save_path, f"whole-{img_id}.png"))
                if "whole" in args.cond_type_test:
                    batch["whole"] = batch["whole"][0].unsqueeze(0)
                else:
                    batch["whole"] = (torch.ones_like(batch["whole"][0].unsqueeze(0)) * (-1)).to(unet.device)
                
            controlnet_image = []
            for key in ['depth', 'midas_depth', 'normal', 'canny', 'body', 'face', 'hand', 'whole']:
                if key in args.cond_type:
                    controlnet_image.append(batch[key][i_batch])
                
            with torch.autocast("cuda"):
                output = pipeline(
                    batch["text_raw"][i_batch], 
                    image=controlnet_image, 
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=500, 
                    generator=generator,
                    negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                )

            image = output.images[0]
            image.save(os.path.join(save_path, f"{img_id}.png"))
                
        if i_test >= 15:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
