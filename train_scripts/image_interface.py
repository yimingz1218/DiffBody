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
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
import time
import matplotlib.pyplot as plt

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
from pipelines.pipeline_stable_diffusion_spade import StableDiffusionPipeline
from collections import OrderedDict
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    ################################### newly added args ###################################
    # parser.add_argument('--prediction_type', type=str, default='v_prediction', choices=['epsilon', 'v_prediction', 'target'], help='Select a mode')
    parser.add_argument("--distill_path", type=str, default=None)
    parser.add_argument('--image-path', default='/fsx/laion/data/openprompts.csv', type=str)
    parser.add_argument('--overfit', default=False, action="store_true")
    parser.add_argument('--copy_weight', default=False, action="store_true")
    parser.add_argument('--copy_weight_same', default=False, action="store_true")
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
    
    img = PIL.Image.open(args.image_path)
    # img.save('/fsx_laion/alvin/visualization/image_interface/pred/tmp1.png')
    img = img.rotate(-90, expand=True)
    img = img.convert("RGB")
    # img.save('/fsx_laion/alvin/visualization/image_interface/pred/tmp2.png')
    # exit(0)
    # img = img.resize((768, 1024))
    original_height = 1024
    original_width = int(img.width * 1024. / img.height) // 8 * 8
    with torch.no_grad():
        # normal_img_tensor = normal_trans_totensor(img)[:3].unsqueeze(0).to(device)
        
        # if normal_img_tensor.shape[1] == 1:
        #     normal_img_tensor = normal_img_tensor.repeat_interleave(3,1)
        
        # normal_output = normal_model(normal_img_tensor).clamp(min=0, max=1)
        # print(normal_output)
        # exit(0)
        # # normal_output = F.interpolate(normal_output, (original_height, original_width), mode='bicubic').to('cuda')
        # trans_back = transforms.Compose([
        #                 transforms.ToPILImage(),
        #                 transforms.Resize((original_height, original_width), interpolation=PIL.Image.BILINEAR),
        #                 ])
        # normal_pil = trans_back(normal_output[0])
        # normal_pil.save(os.path.join('/fsx_laion/alvin/visualization/image_interface/pred/', f"normal.png"))
        # exit(0)
        # image_file = io.BytesIO()
        # # Convert the image to the desired format and save it to the file-like object
        # normal_pil = normal_pil.convert('RGB')
        # normal_pil.save(image_file, format='JPEG')
        # with io.BytesIO(image_file.getvalue()) as stream:
        #     normal_output = PIL.Image.open(stream)
        #     normal_output.load()
        #     normal_output = normal_output.convert("RGB")
        # normal_output = transforms.ToTensor()(normal_output).unsqueeze(0).to('cuda')
        
        canny = np.array(img)

        low_threshold = 100
        high_threshold = 200

        import cv2
        canny = cv2.Canny(canny, low_threshold, high_threshold)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny = Image.fromarray(canny)
        canny = transforms.ToTensor()(canny).unsqueeze(0)
        canny = F.interpolate(canny, (original_height, original_width), mode='bicubic')
        canny = canny.to('cuda')
        # batch["canny"] = canny * 2.0 - 1
        
        depth_img_tensor = depth_trans_totensor(img)[:3].unsqueeze(0).to(device)
        if depth_img_tensor.shape[1] == 1:
            depth_img_tensor = depth_img_tensor.repeat_interleave(3,1)

        depth_output = depth_model(depth_img_tensor).clamp(min=0, max=1)
        depth_output = F.interpolate(depth_output.unsqueeze(0), (original_height, original_width), mode='bicubic').squeeze(0)
        depth_output = depth_output.clamp(0, 1)
        depth_output = 1 - depth_output
        image_file = io.BytesIO()
        plt.imsave(image_file, depth_output.detach().cpu().squeeze(), cmap='viridis')
        with io.BytesIO(image_file.getvalue()) as stream:
            depth_output = PIL.Image.open(stream)
            depth_output.load()
            depth_output = depth_output.convert("RGB")
        depth_output = transforms.ToTensor()(depth_output).unsqueeze(0).to('cuda')
        # print(normal_output.shape, depth_output.shape)
        # exit(0)
        
        batch = {}
        batch["depth"] = depth_output * 2.0 - 1
        batch["canny"] = canny * 2.0 - 1
        # batch["depth"] = (torch.ones_like(batch["canny"]) * (-1)).to('cuda')
        # batch["hand"] = (torch.ones_like(batch["depth"]) * (-1)).to('cuda')
        batch["body"] = (torch.ones_like(batch["depth"]) * (-1)).to('cuda')
        batch["normal"] = (torch.ones_like(batch["depth"]) * (-1)).to('cuda')
        batch["face"] = (torch.ones_like(batch["depth"]) * (-1)).to('cuda')
        
        # batch["normal"] = normal_output * 2.0 - 1
        
        depth_denormalize = (batch["depth"] + 1) / 2.0
        depth_numpy = depth_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        depth_numpy = (depth_numpy * 255).round().astype("uint8")
        depth_pil = Image.fromarray(depth_numpy)
        depth_pil.save(os.path.join('/fsx_laion/alvin/visualization/image_interface/pred/', f"depth.png"))
        
        canny_denormalize = (batch["canny"] + 1) / 2.0
        canny_numpy = canny_denormalize.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        canny_numpy = (canny_numpy * 255).round().astype("uint8")
        canny_pil = Image.fromarray(canny_numpy)
        canny_pil.save(os.path.join('/fsx_laion/alvin/visualization/image_interface/pred/', f"canny.png"))
        # exit(0)
        
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(args.distill_path, subfolder="unet_ema")
        unet.requires_grad_(False)
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
        )
        
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.set_timesteps(500)
        pipeline.scheduler.config.prediction_type = args.prediction_type
        pipeline.scheduler.config['prediction_type'] = args.prediction_type
        pipeline.set_progress_bar_config(disable=False)

        if args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        generator = torch.Generator(device='cuda').manual_seed(0)
        pipeline = pipeline.to('cuda')
        
        os.makedirs('/fsx_laion/alvin/visualization/image_interface/pred/', exist_ok=True)
        images = []
        for i in range(len(args.validation_prompts)):
            with torch.autocast("cuda"):
                output = pipeline(
                    "an anime image of a man", 
                    negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                    num_inference_steps=500, 
                    height=original_height,
                    width=original_width,
                    generator=generator, 
                    batch=batch, 
                    args=args, 
                )
                image = output.images[0]
                image.save(os.path.join('/fsx_laion/alvin/visualization/image_interface/pred/', f"1.png"))

if __name__ == "__main__":
    main()