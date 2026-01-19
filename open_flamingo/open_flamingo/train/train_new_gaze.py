
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
from data import get_data, get_custom_dataset, get_dataset1, get_dataset2,get_dataset3,get_dataset_new,get_dataset_gaze
from distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from train_utils import (
    validate,
    train_one_epoch_gaze
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
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        #osd= checkpoint['optimizer_state_dict']
        #osd = FSDP.optim_state_dict_to_load(osd, model, optimizer)
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch=1
    return start_epoch



def load_checkpoint_gaze(load_path,model,optimizer=None,scheduler=None,rank=0):
    if os.path.isfile(load_path):
        print("loading checkpoint '{}'".format(load_path))
        checkpoint= torch.load(load_path,map_location='cpu')
        FSDP.set_state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=False),
                optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True))
        
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        osd= checkpoint['optimizer_state_dict']
        osd = FSDP.optim_state_dict_to_load(osd, model, optimizer)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = checkpoint['scheduler_state_dict']
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch=1
    return start_epoch



def save_losses_to_file(losses, filepath):
    with open(filepath, 'a') as f:
        for epoch, loss in enumerate(losses, 1):
            f.write(f"Epoch {epoch}: {loss}\n")



def save_loss_to_file(loss, filepath):
    with open(filepath, 'a') as f:
        f.write(f"Epoch {epoch}: {loss}\n")

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

    train_dataset_info = get_dataset_gaze(tokenizer, clip_processor, 'train', rank, world_size)
    train_dataset = train_dataset_info.dataloader
    # valid_dataset_info = get_dataset_new(tokenizer, clip_processor, 'validation', rank, world_size)
    # valid_dataset = valid_dataset_info.dataloader
        # Initialize gradient checkpointing

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
    checkpoint_path = os.path.join("insert path/open_flamingo/open_flamingo/models/gaze_checkpoint.pth")
    if os.path.isfile(checkpoint_path):
        dist.barrier()
        start_epoch = load_checkpoint_new("insert path/open_flamingo/open_flamingo/models/gaze_checkpoint.pth",model,optimizer,rank=rank)
        print(f"Rank {rank}: Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 1
    
    print(f"training starting from {start_epoch}")
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # print("training started")
    for epoch in range(start_epoch, args.epochs + 1):
    #training 
        print(f"new epoch started")
        trainloss= train_one_epoch_gaze(args, model, epoch, train_dataset, tokenizer,optimizer, scheduler, rank, None)  # Assuming wandb or logging handled within
        train_loss.append(trainloss)
    #validation 
        # model.eval()
        # valid_loss = validate(args=args,model=model,valid_loader=valid_dataset,tokenizer=tokenizer,device_id=rank)
        # validation_loss.append(valid_loss)

        
        # if valid_loss < best_valid_loss:
        #     #save this model 
        #     best_valid_loss = validation_loss
        #     best_model_state = model.state_dict()  # Store the best model's state dict

        # if torch.distributed.get_rank() == 0:
        print("saving started")
        save_checkpoint(model, optimizer, scheduler, epoch, "insert path/open_flamingo/open_flamingo/models/gaze_checkpoint.pth")
        save_losses_to_file(train_loss,f"{args.save_dir}/final_train_loss.txt")
    # At the end of training, save the best model's state dict
    # if torch.distributed.get_rank() == 0:
    #     if best_model_state is not None:  # Ensure there was at least one validation
    #         torch.save(best_model_state, f"{args.save_dir}/best_model_checkpoint.pth")
    #         print(f"Best model saved with validation loss: {best_valid_loss}")
        

        
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
    parser.add_argument("--port", type=str, default="1025", help="Distributed training port")
    parser.add_argument("--clip_vision_encoder_path", type=str, default="ViT-L-14")
    parser.add_argument("--clip_vision_encoder_pretrained", type=str, default="openai")
    parser.add_argument("--save_dir",type=str,default="insert path/open_flamingo/open_flamingo/models")
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


