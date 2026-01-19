
import functools
import io
import json
import math
import re
import random
import numpy as np
import torch
import torchvision
import webdataset as wds
from PIL import Image
import base64
from scipy.optimize import linear_sum_assignment
import math
import functools
import webdataset as wds
from open_flamingo import create_model_and_transforms
from io import BytesIO

# from your_preprocess_fn import your_preprocess_fn  # You need to implement this

from data_utils import *

Image.MAX_IMAGE_PIXELS = 1000000000
N_CHANNELS = 3
MIN_KB = 10
_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

# #my preprocess_function
def preprocess_interleaved_custom(
    sample,
    tokenizer,
    clip_processor,
    max_num_images=5,  # Assuming each sequence has exactly 5 images
    max_tokens=256,
):
    """
    Preprocess a sequence of 5 images and pair them with annotations for 2 future actions.
    """
    # info = json.loads(sample[0])  # Loading the sample, which includes images and annotations
    info = json.loads(sample['json']) 

    # Initialize lists to hold processed images and their indices
    images = []
    # for i, image_info in enumerate(info['image_info'][:max_num_images]):
    for image_info in info['images']:
        image_base64 = image_info['image_base64']
        rawbytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
        processed_image = clip_processor(image)  # Process image
        images.append(processed_image)

    # Ensure we have 5 processed images
    if len(images) < max_num_images:
        raise ValueError(f"Found {len(images)} images, expected {max_num_images}")

    # Preprocess images into tensors
    images_tensors = torch.stack(images)  # Assuming clip_processor returns tensors

    # Concatenate annotations for future actions
    text = " ".join(info['annotations'])  # Joining the 2 future action annotations



    text_tensor = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    return images_tensors, (text_tensor["input_ids"], text_tensor["attention_mask"])



#My dataset 
def get_custom_dataset(args, split, image_processor, tokenizer, epoch=0):
    """
    Initialize webdataset for a custom dataset.
    
    Parameters:
    - args: Command line arguments containing paths to the dataset and other configurations.
    - split: The dataset split to load ('train', 'val', or 'test').
    - image_processor: Function to process images.
    - tokenizer: Tokenizer for text processing.
    - epoch: Current epoch for deterministic shuffling.
    """
    if split == 'train':
        input_shards = '/mnt/lv1/ego4d/v2/gazevlm_data/dataset_new/train.tar'
    elif split == 'val':
        input_shards = '/mnt/lv1/ego4d/v2/gazevlm_data/dataset_new/validation.tar'
    elif split == 'test':
        input_shards = '/mnt/lv1/ego4d/v2/gazevlm_data/dataset_new/test.tar'
    else:
        raise ValueError(f"Invalid dataset split: {split}")
    
    assert input_shards is not None, f"Input shards for {split} not specified."

    preprocess_fn = functools.partial(
        preprocess_interleaved,  # This needs to be defined by you.
        clip_processor=image_processor,
        tokenizer=tokenizer,
    )
    # pipeline = [wds.SimpleShardList(input_shards)]

    # dataset = wds.WebDataset(input_shards).decode("pil").map_dict(json=lambda x: json.loads(x.decode('utf-8')))
    # dataset = dataset.map(preprocess_fn)

    # Define your data pipeline
    pipeline = [
        wds.SimpleShardList(input_shards),
        # wds.WebDataset(input_shards),
        #wds.WebDaSimpleShardListtaset(input_shards),
        #wds.shuffle(10000, initial=10000),
        wds.decode('pil'),  # Decodes images using PIL; adjust as needed.
        # wds.to_tuple("jpg;png", "json"),  # Adjust based on your data format.
        wds.map(preprocess_fn),
        wds.batched(args.batch_size, partial=False),
    ]

    dataset = wds.DataPipeline(*pipeline)



    num_samples = 67593
    global_batch_size = 32*4
    num_batches = math.ceil(num_samples / global_batch_size)
    dataset = dataset.with_epoch(num_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=epoch)





def preprocess(sample, clip_processor, tokenizer, max_images=5, max_tokens=256):
    """
    Decode JSON, base64 images, and process annotations for a single sample.
    """
    # Decode the JSON data for a single sample
    json_data = json.loads(sample['json'].decode('utf-8'))

    images = []
    annotations = json_data['annotations']

    # Decode and process each image
    # for img_data in json_data['images'][:max_images]:
    #     image_base64 = img_data
    #     image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
    #     processed_image = clip_processor(image)  # Convert image to tensor
    #     images.append(processed_image)
    for i in range(0,5):
        image_base64 = json_data['images'][i]
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
        processed_image = clip_processor(image)  # Convert image to tensor
        images.append(processed_image)

    if len(images) < max_images:
        raise ValueError(f"Expected {max_images} images, but found {len(images)}")

    # Stack images to create a single tensor
    images_tensor = torch.stack(images)
    

    cleaned_annotations = [re.sub(r'^\d+\.\s*', '', annotation) for annotation in annotations]
    # Concatenate annotations and tokenize
    annotations_text = ' '.join(cleaned_annotations)
    #Here we add the special tokens to ensure that the format is consistent with what flamingo is accepting 

    image_tokens = " ".join(["<image>"] * max_images)

    combined_text = f"{image_tokens}{annotations_text}<|endofchunk|>{tokenizer.eos_token}"
    #print(combined_text)
    annotations_tokenized = tokenizer(
        combined_text,
        max_length=max_tokens,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    annotations_tokenized['input_ids'] = annotations_tokenized['input_ids'].reshape(256)
    annotations_tokenized['attention_mask'] = annotations_tokenized['attention_mask'].reshape(256)
    images_tensor = images_tensor.unsqueeze(1)

    # annotations_tokenized['input_ids'] = (annotations_tokenized['input_ids']).squeeze(1)
    # annotations_tokenized['attention_mask'] = (annotations_tokenized['attention_mask']).squeeze(1)
    return images_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']




def get_dataset1(args,tokenizer,clip_processor,split="test"):
    """
    Initialize a webdataset for the custom dataset.
    """
    # Define the path to the shards based on the split
    input_shards = '/mnt/lv1/ego4d/v2/gazevlm_data/dataset_new/modified/train_{0001-6760}.tar'
    assert input_shards is not None, f"Path to {split} shards must be specified."


    num_samples = 6760
    global_batch_size = 32
    num_batches = math.ceil(num_samples / global_batch_size)
    

    # Use functools.partial to prepare the actual preprocess function with any required arguments
    preprocess_fn = functools.partial(preprocess,  # This needs to be defined by you.
        clip_processor=clip_processor,
        tokenizer=tokenizer,)

    # # Initialize the WebDataset
    # shardlist = wds.ShardList(input_shards, shuffle=(split == "train"), seed=args.seed)

    # # Filter shards for this worker
    # shardlist = input_shards.shards(0, 3)
    dataset = wds.WebDataset(input_shards,nodesplitter=wds.split_by_node)
    
    # Apply preprocessing to each item
    dataset = dataset.map(preprocess_fn)

    # Create a DataLoader
    dataloader = wds.WebLoader(
        dataset,
        batch_size=32,  # Ensure you have args.batch_size defined
        num_workers=1,    # Ensure you have args.workers defined
        shuffle=(split=="test")     # Shuffle only training data
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    print('sample done ')

    return DataInfo(dataloader=dataloader, shared_epoch=0)


def get_dataset2(args, tokenizer, clip_processor, split="test"):
    """
    Initialize a webdataset for the custom dataset.
    """
    # Adjusting the pattern for input shards
    if split == "train":
        input_shards = f'{args.train_shards}/train_{{000..6759}}.tar'  # Adjusted pattern
    elif split == "val":
        input_shards = f'{args.val_shards}/validation_{{000..1690}}.tar'  # Update XXX with the correct range
    elif split == "test":
        input_shards = f'{args.test_shards}/test_{{000..2113}}.tar'  # Update XXX with the correct range
    else:
        raise ValueError(f"Invalid dataset split: {split}")

    assert os.path.exists(os.path.dirname(input_shards)), f"Input shards path for {split} does not exist."

    num_samples = {'train': 6760, 'val': 1690, 'test': 2113}[split]  # Update XXX with actual numbers
    global_batch_size = 32
    num_batches = math.ceil(num_samples / global_batch_size)

    preprocess_fn = functools.partial(preprocess, clip_processor=clip_processor, tokenizer=tokenizer)

    # Create the dataset
    dataset = wds.WebDataset(input_shards, nodesplitter=wds.split_by_node).map(preprocess_fn)

    # Create a DataLoader
    dataloader = wds.WebLoader(dataset, batch_size=32, num_workers=args.workers, shuffle=(split == "test"))

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    print('Dataset setup complete')
    return DataInfo(dataloader=dataloader, shared_epoch=0)



def generate_shard_list(split, base_path, start_index, end_index):
    """Generate formatted shard paths for a given range."""
    # Adjust format based on how your files are named (e.g., 'train_000.tar')
    shard_list = [f"{base_path}/{split}_{i:03d}.tar" for i in range(start_index, end_index + 1)]
    return shard_list

def get_dataset_new(tokenizer, clip_processor, split="train", rank=0, world_size=1):
    # Define base path and total number of samples per split
    base_path = '/mnt/lv1/ego4d/v2/gazevlm_data/dataset_new/modified'
    num_samples_per_split = {
        "train": 6759,
        "validation": 1689,
        "test": 2113,
    }

    # Define start and end index for each split based on your file naming
    shard_indices = {
        "train": (0, 6759),  # Adjust these values based on your actual file range
        "validation": (0, 1689),  # Adjust these values based on your actual file range
        "test": (0, 2113),   # Adjust these values based on your actual file range
    }

    num_samples = num_samples_per_split[split]
    global_batch_size = 32 * world_size
    num_batches = math.ceil(num_samples / global_batch_size)

    preprocess_fn = functools.partial(preprocess,
        clip_processor=clip_processor,
        tokenizer=tokenizer)

    # Generate shard list based on the specified range
    shard_list = generate_shard_list(split, base_path, *shard_indices[split])

    # Create the WebDataset
    dataset = wds.WebDataset(shard_list).map(preprocess_fn)

    # if world_size > 1:
    #     # In distributed mode, ensure data is split correctly across workers
    #     dataset = dataset.shard(num_shards=world_size, shard_id=rank)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=32,
        num_workers=3,  # Adjust according to your setup
        shuffle=False,  # Shuffle only for training data
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=0)


def main():
    class Args:
        batch_size = 4
        workers = 1  # Set to 0 for debugging purposes to ensure single-threaded execution
    # model, image_processor, text_tokenizer

    args = Args()
    model, clip_processor, tokenizer = create_model_and_transforms(clip_vision_encoder_path="ViT-L-14" ,
   clip_vision_encoder_pretrained="openai",
   lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b")
    # Adjust these paths as necessary
    args.train_shards = '/mnt/lv1/ego4d/v2/gazevlm_data/dataset_new/modified'
    args.val_shards = '/mnt/lv1/ego4d/v2/gazevlm_data/dataset_new/validation.tar'
    args.test_shards = '/mnt/lv1/ego4d/v2/gazevlm_data/dataset_new/test.tar'

    # Choose which split to debug
    dataset_split = 'validation'  # or 'val' or 'test'

    # Get the dataset
    dataset_info = get_dataset_new(split= dataset_split, tokenizer=tokenizer, clip_processor=clip_processor)

    # # Load a few samples and process them
    # for batch_idx, (images, targets) in enumerate(dataset_info.dataloader):
    #     print(f"Batch {batch_idx + 1}")
    #     print(f"Images shape: {images.shape}, Targets shape: {targets[0].shape}")

    #     # Optionally, visualize or further inspect the batch here

    #     if batch_idx >= 1:  # Process only two batches for debugging
    #         break
    # Inspect the first batch
    # first_batch = next(iter(dataset_info.dataloader))
    # print(f"Type of first_batch: {type(first_batch)}")

    # if isinstance(first_batch, tuple):
    #     print(f"Length of tuple: {len(first_batch)}")
    # elif isinstance(first_batch, dict):
    #     print(f"Keys in dict: ")
    # else:
    #     print("Unexpected batch structure:")
    for batch_idx, batch in enumerate(dataset_info.dataloader):
        # images = batch[0]
        # targets = batch[2]  # or adjust if targets are located differently
        # # total_size = len(dataset_info.dataloader)  # Total size of the dataset
        # # batch_size = dataset_info.dataloader.batch_size  # Batch size used in the DataLoader
        # # total_iterations = -(-total_size // batch_size)  # Ceiling division to round up
        # print(f"length is {dataset_info.getsize()}")
        print(f"Batch {batch_idx + 1}")
        # print(f"Images shape: {images.shape}, Targets shape: {targets.shape}")
        # if batch_idx >= 2:  # Process only two batches for debugging
        #     break



if __name__ == "__main__":
    main()


# Train .tar files: 6760
# Test .tar files: 2113
# Validation .tar files: 1690