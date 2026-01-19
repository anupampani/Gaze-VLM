#program to see if I can load model checkpoint successfully

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
from data import get_data, get_custom_dataset, get_dataset1, get_dataset2,get_dataset3,get_dataset_new
from distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from train_utils import (
    validate,
    train_one_epoch_v3,
    train_one_epoch_v2,
    train_one_epoch_new,
    get_mp_policy_dtype,
    save_checkpoint
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

def load_checkpoint_new(load_path,model,optimizer=None,scheduler=None,rank=0):
    if os.path.isfile(load_path):
        print("loading checkpoint '{}'".format(load_path))
        # if rank==0:
        checkpoint= torch.load(load_path,map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        print("checkpoint in between")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch=1
    return start_epoch
# def save_checkpoint(model, optimizer, scheduler, epoch, filepath):
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#     }, filepath)

# def load_checkpoint(model, optimizer, scheduler, filepath):
#     checkpoint = torch.load(filepath)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1  # Continue from the next epoch
#     return start_epoch

def load_checkpoint_simple(model, optimizer, scheduler, filepath, rank):
    """
    Simplified checkpoint loading function for models wrapped with FSDP.

    Args:
        model (torch.nn.Module): The FSDP wrapped model.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        filepath (str): Path to the checkpoint file.
        rank (int): The rank of the current process in distributed training.

    Returns:
        int: The epoch to resume training from.
    """
    # Ensure the file exists
    if rank == 0:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        checkpoint = torch.load(filepath, map_location=f"cuda:{rank}")
    else:
        checkpoint = None
    print("checkpoint1")
    # Use torch.distributed.barrier() to synchronize all processes.
    # This ensures that the checkpoint is loaded by rank 0 before proceeding.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    print("checkpoint2")
    # Prepare a list for broadcast. The checkpoint will be the first item for rank 0,
    # and an empty dict for other ranks.
    checkpoint_list = [checkpoint] if rank == 0 else [None]

    # Broadcast the checkpoint from rank 0 to all other ranks.
    torch.distributed.broadcast_object_list(checkpoint_list, src=0)
    print("checkpoint3")
    # After broadcast, all ranks will have the checkpoint in checkpoint_list[0].
    checkpoint = checkpoint_list[0]
    print("checkpoint4")
    if checkpoint is not None:
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 1

    # Use another barrier to ensure all processes have loaded the checkpoint
    # before moving forward.
    print("checkpoint5")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return start_epoch


def save_losses_to_file(losses, filepath):
    with open(filepath, 'w') as f:
        for epoch, loss in enumerate(losses, 1):
            f.write(f"Epoch {epoch}: {loss}\n")

    
def fsdp_main(rank, world_size, args):
    setup_distributed(args.port, rank, world_size)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(rank)
    random_seed(args.seed, rank)

    model, clip_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=args.clip_vision_encoder_path,
        clip_vision_encoder_pretrained=args.clip_vision_encoder_pretrained,
        lang_encoder_path=args.lang_encoder_path,
        gradient_checkpointing=True,
        tokenizer_path=args.tokenizer_path,
        freeze_lm_embeddings=True,
    )

    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        # offload_to_cpu=True,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
        and not isinstance(m, FSDP)
        and not isinstance(m, CheckpointWrapper),
    )

    model = model.to(f"cuda:{rank}")
    wrapper_kwargs = dict(
        process_group=None,
        cpu_offload=CPUOffload(offload_params=False),
        device_id=rank,
        sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        # if args.fsdp_sharding_strategy == "full"
        # else ShardingStrategy.HYBRID_SHARD,
        use_orig_params=True,
        # mixed_precision=mp_policy,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )
    model.wrap_fsdp(wrapper_kwargs, rank)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


    best_valid_loss = float('inf') 
    train_loss=[]
    train_accuracy=[]
    validation_loss=[]
    # Check if a checkpoint exists and load it
    checkpoint_path = os.path.join("insert path/open_flamingo/open_flamingo/models/last_checkpoint.pth")
    if os.path.isfile(checkpoint_path):
        print("checkpoint1")
        dist.barrier()  # Ensure all ranks load their checkpoint before proceeding
        print("checkpoint2")
        start_epoch = load_checkpoint_new(checkpoint_path,model, optimizer,rank)
        #print(f"Rank {rank}: Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 1
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print(f"Rank {rank}: Resuming training from epoch {start_epoch}")
    
    cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP Example with WebDataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp_bfloat16",
        help="Floating point precision.",)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=str, default="1025", help="Distributed training port")
    parser.add_argument("--clip_vision_encoder_path", type=str, default="ViT-L-14")
    parser.add_argument("--clip_vision_encoder_pretrained", type=str, default="openai")
    parser.add_argument("--save_dir",type=str,default="insert path/open_flamingo/open_flamingo/model1_a")
    # parser.add_argument("--lang_encoder_path", default="facebook/opt-1.3b", type=str)
    # parser.add_argument("--tokenizer_path", default="facebook/opt-30b", type=str)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lang_encoder_path", type=str, default="anas-awadalla/mpt-1b-redpajama-200b")
    parser.add_argument("--tokenizer_path", type=str, default="anas-awadalla/mpt-1b-redpajama-200b")
    parser.add_argument("--step_size", type=int, default=5, help="Step size for LR scheduler")
    parser.add_argument("--gamma", type=float, default=0.7, help="Gamma for LR scheduler")
    # Include other arguments as needed

    args = parser.parse_args()
    WORLD_SIZE = 4
    #torch.cuda.device_count()
    mp.spawn(fsdp_main,
    args=(WORLD_SIZE, args),
    nprocs=WORLD_SIZE,
    join=True)


