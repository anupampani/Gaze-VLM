import argparse
import os
import webdataset as wds
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from model import MyModel

def setup_distributed(port="29500"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend='nccl')

def cleanup_distributed():
    dist.destroy_process_group()

def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        # Assuming batch contains data and target
        data, target = batch["input"], batch["target"]
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        if dist.get_rank() == 0 and batch_idx % args.log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader) * args.batch_size}] Loss: {loss.item()}")

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
