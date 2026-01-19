#quantify the metric required for evaluation of output generated 


import argparse
import os
import webdataset as wds
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
""" Main training script """

import argparse
import glob
import os
import random

import numpy as np
import torch
import wandb
from data import get_dataset_gaze_eval,get_dataset_gaze,get_small_dataset,get_dataset_new
from distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from train_utils import (
    base_generate_debug,
    text_generate_debug,
    get_mp_policy_dtype,
    save_checkpoint,
    save_checkpoint_gaze
)

from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp._init_utils import _init_intra_and_inter_node_groups
from torch.distributed.distributed_c10d import _get_default_group
import functools
from torch.optim.lr_scheduler import StepLR


import sys
sys.path.insert(0, 'insert path/open_flamingo/')

from open_flamingo import create_model_and_transforms2
import torch.multiprocessing as mp

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"




def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def setup_distributed(port="29500",rank=None,world_size=None):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

def load_checkpoint_eval(load_path):
    checkpoint=None
    if os.path.isfile(load_path):
        
        print("loading checkpoint '{}'".format(load_path))
        checkpoint= torch.load(load_path,map_location='cpu')
    return checkpoint


def save_loss_to_file(loss, filepath):
    with open(filepath, 'a') as f:
        f.write(f"loss: {loss}\n")
    
def fsdp_main(rank, world_size, args):
    setup_distributed(args.port, rank, world_size)
    
    torch.cuda.set_device(rank)
    random_seed(args.seed, rank)

    #model initialization 
    model, clip_processor, tokenizer = create_model_and_transforms2(
        clip_vision_encoder_path=args.clip_vision_encoder_path,
        clip_vision_encoder_pretrained=args.clip_vision_encoder_pretrained,
        lang_encoder_path=args.lang_encoder_path,
        gradient_checkpointing=True,
        tokenizer_path=args.tokenizer_path,
        freeze_lm_embeddings=True,
    )

    #dataset loading for debug evaluation - lets use 31 slices 
    test_dataset_info = get_dataset_new(tokenizer, clip_processor, 'test', rank, world_size)
    test_loader = test_dataset_info.dataloader



    checkpoint_path = os.path.join("insert path/open_flamingo/open_flamingo/models/base_model/base_model_checkpoint_10.pth")
    if os.path.isfile(checkpoint_path):
        dist.barrier()
        checkpoint = load_checkpoint_eval("insert path/open_flamingo/open_flamingo/models/base_model/base_model_checkpoint_10.pth")
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
            #checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        if rank==0:
            model.load_state_dict(checkpoint,strict=False)
    print("model loaded ")
    model = model.to(f"cuda:{rank}")
    dist.barrier()
    test_loss=[]
    print("generate command called")
    text = base_generate_debug(model,test_loader,tokenizer,rank)

    print("over")
    #this is where the evaluation function should be there 
    cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP Example with WebDataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp_bfloat16",
        help="Floating point precision.",)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=str, default="3111", help="Distributed training port")
    parser.add_argument("--clip_vision_encoder_path", type=str, default="ViT-L-14")
    parser.add_argument("--clip_vision_encoder_pretrained", type=str, default="openai")
    parser.add_argument("--save_dir",type=str,default="insert path/open_flamingo/open_flamingo/models")
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    # parser.add_argument("--lang_encoder_path", type=str, default="anas-awadalla/mpt-1b-redpajama-200b")
    # parser.add_argument("--tokenizer_path", type=str, default="anas-awadalla/mpt-1b-redpajama-200b")
    parser.add_argument("--lang_encoder_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument("--tokenizer_path", default="facebook/opt-30b", type=str)
    parser.add_argument("--step_size", type=int, default=10, help="Step size for LR scheduler")
    parser.add_argument("--gamma", type=float, default=0.7, help="Gamma for LR scheduler")
    # Include other arguments as needed

    args = parser.parse_args()
    WORLD_SIZE = 1
    #torch.cuda.device_count()
    mp.spawn(fsdp_main,
    args=(WORLD_SIZE, args),
    nprocs=WORLD_SIZE,
    join=True)
