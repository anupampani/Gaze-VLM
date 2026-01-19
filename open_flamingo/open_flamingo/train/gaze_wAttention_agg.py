
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
from data import get_dataset_gaze, get_dataset_gaze_attention
from distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from train_utils_attention import (
    train_gaze_attention,
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


from open_flamingo import create_model_and_transforms
import torch.multiprocessing as mp

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["WANDB_MODE"]="offline"



def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def setup_distributed(port="26500",rank=None,world_size=None):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

# def save_checkpoint(model, optimizer, scheduler, epoch, filepath):
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#     }, filepath)

def load_checkpoint(model, optimizer, scheduler, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    # osd= checkpoint['optimizer_state_dict']
    # osd = FSDP.optim_state_dict_to_load(osd, model, optimizer)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Continue from the next epoch
    return start_epoch


def load_checkpoint_new(load_path,model,optimizer=None,scheduler=None,rank=0):
    if os.path.isfile(load_path):
        print("loading checkpoint '{}'".format(load_path))
        checkpoint= torch.load(load_path,map_location='cpu')
        FSDP.set_state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=False),
                optim_state_dict_config=FullStateDictConfig(rank0_only=True))
        
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        osd= checkpoint['optimizer_state_dict']
        osd = FSDP.optim_state_dict_to_load(osd, model, optimizer)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch=1
    return start_epoch


def load_checkpoint_gaze(load_path):
    checkpoint=None
    if os.path.isfile(load_path):
        
        print("loading checkpoint '{}'".format(load_path))
        checkpoint= torch.load(load_path,map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch=1
    return start_epoch, checkpoint


def save_loss_to_file(loss, filepath):
    with open(filepath, 'a') as f:
        f.write(f"loss: {loss}\n")


def load_checkpoint_newest(load_path,model,optimizer=None,scheduler=None,rank=0):
    if os.path.isfile(load_path):
        print("loading checkpoint '{}'".format(load_path))
        if rank==0:

            checkpoint= torch.load(load_path,map_location='cpu')
            FSDP.set_state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=False),
                    optim_state_dict_config=FullStateDictConfig(rank0_only=True))
        
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        osd= checkpoint['optimizer_state_dict']
        osd = FSDP.optim_state_dict_to_load(osd, model, optimizer)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch=1
    return start_epoch






def save_losses_to_file(losses, filepath):
    with open(filepath, 'a') as f:
        for epoch, loss in enumerate(losses, 1):
            f.write(f"Epoch {epoch}: {loss}\n")

    
def fsdp_main(rank, world_size, args):
    setup_distributed(args.port, rank, world_size)
    wandb.init(
        project = "gaze_caption_2",
        config = {
            "learning_rate":7e-5,
            "epochs":10,
        }
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(rank)
    random_seed(args.seed, rank)

    #model initialization 
    model, clip_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=args.clip_vision_encoder_path,
        clip_vision_encoder_pretrained=args.clip_vision_encoder_pretrained,
        lang_encoder_path=args.lang_encoder_path,
        gradient_checkpointing=True,
        tokenizer_path=args.tokenizer_path,
        freeze_lm_embeddings=True,
    )

    #dataset loading 
    train_dataset_info = get_dataset_gaze_attention(tokenizer, clip_processor, 'train', rank, world_size,base_path='/home/pani3/dataset_agg')
    train_dataset = train_dataset_info.dataloader

    # valid_dataset_info = get_dataset_gaze_attention(tokenizer, clip_processor, 'validation', rank, world_size)
    # valid_dataset = valid_dataset_info.dataloader


    start_epoch=1

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

    params_to_optimize = model.named_parameters()
    params_to_optimize = list(
        filter(
            lambda x: x[1].requires_grad
            and not getattr(x[1], "exclude_from_optimizer", False),
            params_to_optimize,
        )
    )
    def get_grouped_params(params_to_optimize):
        params_with_wd, params_without_wd = [], []
        for n, p in params_to_optimize:
            if "gated_cross_attn" in n:
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)
        return [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]
    optimizer = optim.Adam(get_grouped_params(params_to_optimize), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    checkpoint_path = os.path.join("insert path/open_flamingo/open_flamingo/models/review2/gaze_model_116.pth")
    if os.path.isfile(checkpoint_path):
        dist.barrier()
        resume_checkpoint=1
        start_epoch ,checkpoint = load_checkpoint_gaze("insert path/open_flamingo/open_flamingo/models/review2/gaze_model_116.pth")
        if rank==0:
            model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        
        print(f"Rank {rank}: Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 1
        resume_checkpoint=0

    wrapper_kwargs = dict(
        process_group=None,
        cpu_offload=CPUOffload(offload_params=False),
        device_id=rank,
        sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )

    print("checkpoint2")
    model.wrap_fsdp(wrapper_kwargs, rank)
    model = model.to(f"cuda:{rank}")
    dist.barrier()
    if resume_checkpoint==1:
        #optimizer and scheduler 
        osd = checkpoint['optimizer_state_dict']
        osd = FSDP.optim_state_dict_to_load(model, optimizer, osd)
        optimizer.load_state_dict(osd)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    train_loss=[]
    best_loss=100000
    print(f"training starting from {start_epoch}")

    # print("training started")
    for epoch in range(start_epoch, args.epochs + 1):
    #training 
        print(f"new epoch started")
        trainloss= train_gaze_attention(args, model, epoch, train_dataset, tokenizer,optimizer, scheduler, rank,wandb)  # Assuming wandb or logging handled within
        train_loss.append(trainloss)
        print("saving started")
        if( trainloss < best_loss):
            best_loss=trainloss
            save_checkpoint_gaze(model, optimizer, scheduler, epoch, f'insert path/open_flamingo/open_flamingo/models/review2/gaze_model.pth')
        if rank==0:
            save_loss_to_file(trainloss,f"{args.save_dir}/gaze_train_loss_115.txt")
                
    cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP Example with WebDataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp_bfloat16",
        help="Floating point precision.",)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=str, default="1087", help="Distributed training port")
    parser.add_argument("--clip_vision_encoder_path", type=str, default="ViT-L-14")
    parser.add_argument("--clip_vision_encoder_pretrained", type=str, default="openai")
    parser.add_argument("--save_dir",type=str,default="insert path/open_flamingo/open_flamingo/models/review2/")
    # parser.add_argument("--lang_encoder_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument("--lang_encoder_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument("--tokenizer_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    # parser.add_argument("--lang_encoder_path", type=str, default="anas-awadalla/mpt-1b-redpajama-200b")
    # parser.add_argument("--tokenizer_path", type=str, default="anas-awadalla/mpt-1b-redpajama-200b")
    parser.add_argument("--step_size", type=int, default=20, help="Step size for LR scheduler")
    parser.add_argument("--gamma", type=float, default=0.7, help="Gamma for LR scheduler")
    # Include other arguments as needed

    args = parser.parse_args()
    WORLD_SIZE = 2
    #torch.cuda.device_count()
    mp.spawn(fsdp_main,
    args=(WORLD_SIZE, args),
    nprocs=WORLD_SIZE,
    join=True)



