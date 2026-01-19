#train function 
import argparse
import os
import webdataset as wds
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from model import MyModel

def setup_distributed(port="29500"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend='nccl')

def cleanup_distributed():
    dist.destroy_process_group()

def train(args, model, custom_loader, optimizer, lr_scheduler, device_id, tokenizer, wandb):
    model.train()
    for epoch in range(args.num_epochs):
        # Call to train one epoch
        train_one_epoch_new(args, model, epoch, custom_loader, tokenizer, optimizer, lr_scheduler, device_id, wandb)
        
        # Optional: Gather loss from train_one_epoch_new if returned
        
        # Synchronize and log loss here
        if dist.get_rank() == 0:
            # Assuming loss_tensor is retrieved or calculated
            # dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            global_loss = loss_tensor.item() / dist.get_world_size()
            print(f"Global Average Loss after Epoch {epoch+1}: {global_loss}")
            wandb.log({"global_loss_epoch": global_loss, "epoch": epoch+1})


# def train(args, model, custom_loader, optimizer, lr_scheduler, device_id, tokenizer, wandb):
#     model.train()
#     for epoch in range(args.num_epochs):
#         local_loss = train_one_epoch_new(args, model, epoch, custom_loader, tokenizer, optimizer, lr_scheduler, device_id, wandb)
        
#         # Synchronize and compute global average loss
#         loss_tensor = torch.tensor(local_loss).to(device_id)
#         dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
#         global_loss = loss_tensor.item() / dist.get_world_size()
        
#         if dist.get_rank() == 0:
#             print(f"Global Average Loss after Epoch {epoch+1}: {global_loss}")
#             wandb.log({"global_loss_epoch": global_loss, "epoch": epoch+1})


# def train(args, model, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, batch in enumerate(train_loader):
#         # Assuming batch contains data and target
#         data, target = batch["input"], batch["target"]
#         data, target = data.cuda(), target.cuda()

#         optimizer.zero_grad()
#         output = model(data)
#         loss = nn.CrossEntropyLoss()(output, target)
#         loss.backward()
#         optimizer.step()

#         if dist.get_rank() == 0 and batch_idx % args.log_interval == 0:
#             print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader) * args.batch_size}] Loss: {loss.item()}")

def main():
    parser = argparse.ArgumentParser(description="PyTorch FSDP Example with WebDataset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-url", type=str, required=True, help="URL/path to WebDataset")
    # Add more arguments as needed
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    setup_distributed()

    # Setup WebDataset
    dataset = wds.WebDataset(args.data_url).decode("pil").to_tuple("jpg;png", "cls").batched(args.batch_size)
    train_loader = wds.WebLoader(dataset, num_workers=4, batch_size=None).ddp_equalize(len(dataset))

    model = MyModel().cuda()
    model = FSDP(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)

    if dist.get_rank() == 0:
        # Save model checkpoint
        torch.save(model.state_dict(), "model_checkpoint.pth")

    cleanup_distributed()

if __name__ == "__main__":
    main()

