import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Simple model definition
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def init_distributed_mode():
    # Make sure to set these environment variables before running your script
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"Initialized process group; rank: {dist.get_rank()}, size: {dist.get_world_size()}, device: {local_rank}")

def setup():
    init_distributed_mode()
    model = SimpleModel().cuda()
    fsdp_model = FSDP(model)
    print(f"Model is wrapped with FSDP: {fsdp_model}")

    # Dummy data
    input_data = torch.randn(64, 10).cuda()
    output_data = torch.randn(64, 1).cuda()
    return fsdp_model, input_data, output_data

def train(model, input_data, output_data):
    print("yolo1")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(5):  # simple 5 epoch training
        print("yolo-inside epoch")
        optimizer.zero_grad()
        print("yolo-2-inside epoch")
        outputs = model(input_data)
        loss = criterion(outputs, output_data)
        loss.backward()
        optimizer.step()

    #     if dist.get_rank() == 0:  # Only print from one process
    #         print(f"Epoch {epoch}, Loss: {loss.item()}")

    # # Ensure all processes have finished training
    # # dist.barrier()
    # if dist.get_rank() == 0:
    #     print("Training completed successfully.")

if __name__ == "__main__":
    model, input_data, output_data = setup()
    train(model, input_data, output_data)
    dist.barrier()  # Ensure all processes reach this point
    if dist.get_rank() == 0:
        print("Training completed successfully.")
    torch.cuda.synchronize()  # Ensure CUDA operations have completed
    dist.destroy_process_group()  # Cleanly destroy the process group
