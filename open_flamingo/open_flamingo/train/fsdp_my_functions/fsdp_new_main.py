
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
from data import get_data, get_custom_dataset, get_dataset1
from distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from train_utils import (
    train_one_epoch_new,
    get_mp_policy_dtype,
    save_checkpoint,
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

from open_flamingo import create_model_and_transforms
import torch.multiprocessing as mp

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


#my train.py
# def main():
#     parser = argparse.ArgumentParser(description="PyTorch FSDP Example with WebDataset")
#     parser.add_argument("--batch-size", type=int, default=64)
#     parser.add_argument("--epochs", type=int, default=14)
#     parser.add_argument("--lr", type=float, default=0.01)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--data-url", type=str, required=True, help="URL/path to WebDataset")
#     # Add more arguments as needed
#     args = parser.parse_args()

#     torch.manual_seed(args.seed)
#     setup_distributed()
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()
    
#     # Initialize model, tokenizer, and clip_processor
#     model, clip_processor, tokenizer = create_model_and_transforms(
#         clip_vision_encoder_path="ViT-L-14",
#         clip_vision_encoder_pretrained="openai",
#         lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
#         tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b"
#     )
    
#     # Adjust get_dataset1 to accept rank and world_size if necessary for partitioning
#     custom_dataset_info = get_dataset1(tokenizer, clip_processor,'train', rank, world_size)
#     custom_loader = custom_dataset_info.dataloader  # Assuming this returns a WebLoader

#     # Move model to device and wrap with FSDP
#     model = model.to(rank)
#     model = FSDP(model)
    
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

#     for epoch in range(1, args.epochs + 1):
#         train(args, model, custom_loader, optimizer, epoch)

#     if dist.get_rank() == 0:
#         # Save model checkpoint
#         torch.save(model.state_dict(), "model_checkpoint.pth")

#     cleanup_distributed()

# if __name__ == "__main__":
#     args = ...  # Your argument parsing here
#     world_size = torch.cuda.device_count()
#     mp.spawn(main, args=(world_size, args,), nprocs=world_size, join=True)
    
def fsdp_main(rank, world_size, args):
    setup_distributed(args.port, rank, world_size)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    random_seed(args.seed, rank)

    model, clip_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=args.clip_vision_encoder_path,
        clip_vision_encoder_pretrained=args.clip_vision_encoder_pretrained,
        lang_encoder_path=args.lang_encoder_path,
        tokenizer_path=args.tokenizer_path
    )

    custom_dataset_info = get_dataset1(tokenizer, clip_processor, 'train', rank, world_size)
    custom_loader = custom_dataset_info.dataloader

    model = model.cuda(rank)
    wrapper_kwargs = dict(
        process_group=None,
        cpu_offload=CPUOffload(offload_params=False),
        device_id=rank,
        sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        # if args.fsdp_sharding_strategy == "full"
        # else ShardingStrategy.HYBRID_SHARD,
        use_orig_params=args.fsdp_use_orig_params,
        # mixed_precision=mp_policy,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )
    model.wrap_fsdp(wrapper_kwargs, rank)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch_new(args, model, epoch, custom_loader, optimizer, scheduler, rank, tokenizer, None)  # Assuming wandb or logging handled within

    if rank == 0:
        torch.save(model.state_dict(), "model_checkpoint.pth")

    cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP Example with WebDataset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=str, default="29500", help="Distributed training port")
    parser.add_argument("--clip_vision_encoder_path", type=str, default="ViT-L-14")
    parser.add_argument("--clip_vision_encoder_pretrained", type=str, default="openai")
    parser.add_argument("--lang_encoder_path", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--tokenizer_path", type=str, default="facebook/opt-30b")
    parser.add_argument("--step_size", type=int, default=1, help="Step size for LR scheduler")
    parser.add_argument("--gamma", type=float, default=0.7, help="Gamma for LR scheduler")
    # Include other arguments as needed

    args = parser.parse_args()
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
    args=(WORLD_SIZE, args),
    nprocs=WORLD_SIZE,
    join=True)