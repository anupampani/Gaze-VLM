"""
Preprocess and load datasets for training.
"""
import random
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
from io import BytesIO
import random
# from your_preprocess_fn import your_preprocess_fn  # You need to implement this
import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
# from transformers import AutoModel, AutoTokenizer
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




import random
import numpy as np
from PIL import ImageDraw, ImageFilter

def preprocess_gaze_dist2(sample, clip_processor, tokenizer, max_images=5, max_tokens=256, corruption_probability=1.0):
    """
    Decode JSON, base64 images, and process annotations for a single sample.
    Introduces corruption to gaze maps and overlaid images.
    """
    # Decode the JSON data for a single sample
    json_data = json.loads(sample['json'].decode('utf-8'))

    images = []
    gaze = []
    overlaid = []
    annotations = json_data['annotations']

    # Define a blank image to use as corrupted data
    blank_image = Image.new('RGB', (224, 224), color=(0, 0, 0))  # Black blank image
    random_image = Image.new('RGB', (224, 224), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    def add_random_bright_spots(image, num_spots=10, spot_size=5):
        """
        Add random bright spots to an image to simulate corrupted gaze maps.
        """
        draw = ImageDraw.Draw(image)
        for _ in range(num_spots):
            x = random.randint(0, image.width - spot_size)
            y = random.randint(0, image.height - spot_size)
            box = [x, y, x + spot_size, y + spot_size]
            draw.rectangle(box, fill=(255, 255, 255))  # Bright white spots
        return image

    def corrupt_overlay(image):
        """
        Corrupt overlay with noise or random cropping and resizing.
        """
        if random.random() < 0.5:
            # Add random noise
            np_image = np.array(image)
            noise = np.random.randint(0, 50, np_image.shape, dtype=np.uint8)
            corrupted = Image.fromarray(np.clip(np_image + noise, 0, 255).astype(np.uint8))
        else:
            # Random crop and resize
            crop_size = random.randint(100, 200)
            cropped = image.crop((0, 0, crop_size, crop_size))
            corrupted = cropped.resize((224, 224))
        return corrupted

    # Decode and process each image
    for i in range(0, max_images):
        image_base64 = json_data['images'][i]
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
        processed_image1 = clip_processor(image)  # Convert image to tensor

        # Randomly corrupt gaze maps
        heatmap_base64 = json_data['gazemaps'][i]
        heatmap = Image.open(BytesIO(base64.b64decode(heatmap_base64))).convert('RGB')
        if random.random() < corruption_probability:
            heatmap = add_random_bright_spots(blank_image.copy())
        processed_image2 = clip_processor(heatmap)  # Convert image to tensor

        # Randomly corrupt overlays
        overlay_base64 = json_data['overlays'][i]
        overlays = Image.open(BytesIO(base64.b64decode(overlay_base64))).convert('RGB')
        if random.random() < corruption_probability:
            overlays = corrupt_overlay(overlays)
        processed_image3 = clip_processor(overlays)  # Convert image to tensor

        gaze.append(processed_image2)
        overlaid.append(processed_image3)
        images.append(processed_image1)

    if len(images) < max_images:
        raise ValueError(f"Expected {max_images} images, but found {len(images)}")

    # Stack images to create a single tensor
    images_tensor = torch.stack(images)
    gaze_tensor = torch.stack(gaze)
    overlaid_tensor = torch.stack(overlaid)

    clean_annotations = []
    for annotation in annotations:
        # Remove the pattern "number dot space" using regular expressions
        cleaned_annotation = re.sub(r'\d+\.\s', '', annotation)
        clean_annotations.append(cleaned_annotation)

    annotations_text = ''.join(clean_annotations)
    image_tokens = " ".join(["<image>"] * max_images)
    combined_text = f"{image_tokens}{annotations_text}<|endofchunk|>{tokenizer.eos_token}"
    # tokenizer.padding_side="left"
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
    gaze_tensor = gaze_tensor.unsqueeze(1)
    overlaid_tensor = overlaid_tensor.unsqueeze(1)

    del images  # Explicitly delete the list to free up memory
    torch.cuda.empty_cache()  # Use sparingly
    return images_tensor, overlaid_tensor, gaze_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']


def preprocess_gaze_dist(sample, clip_processor, tokenizer, max_images=5, max_tokens=256, corruption_probability=0.6):
    """
    Decode JSON, base64 images, and process annotations for a single sample.
    Randomly corrupts gaze maps and overlaid images with a given probability.
    """
    # Decode the JSON data for a single sample
    json_data = json.loads(sample['json'].decode('utf-8'))

    images = []
    gaze = []
    overlaid = []
    annotations = json_data['annotations']

    # Define a blank image to use as corrupted data
    blank_image = Image.new('RGB', (224, 224), color=(0, 0, 0))  # Replace size and color as needed

    # Decode and process each image
    for i in range(0, max_images):
        image_base64 = json_data['images'][i]
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
        processed_image1 = clip_processor(image)  # Convert image to tensor

        # Randomly corrupt gaze maps and overlays
        # if random.random() < 1.0:
        #     heatmap = blank_image  # Replace with blank image
        # else:
        heatmap_base64 = json_data['gazemaps'][i]
        heatmap = Image.open(BytesIO(base64.b64decode(heatmap_base64))).convert('RGB')
        processed_image2 = clip_processor(heatmap)  # Convert image to tensor

        # if random.random() < corruption_probability:
        #     # overlays = blank_image  # Replace with blank image
        #     overlay_base64 = json_data['images'][i]
        # else:
        overlay_base64 = json_data['images'][i]
        
        overlays = Image.open(BytesIO(base64.b64decode(overlay_base64))).convert('RGB')
        processed_image3 = clip_processor(overlays)  # Convert image to tensor

        gaze.append(processed_image2)
        overlaid.append(processed_image3)
        images.append(processed_image1)

    if len(images) < max_images:
        raise ValueError(f"Expected {max_images} images, but found {len(images)}")

    # Stack images to create a single tensor
    images_tensor = torch.stack(images)
    gaze_tensor = torch.stack(gaze)
    overlaid_tensor = torch.stack(overlaid)

    clean_annotations = []
    for annotation in annotations:
        # Remove the pattern "number dot space" using regular expressions
        cleaned_annotation = re.sub(r'\d+\.\s', '', annotation)
        clean_annotations.append(cleaned_annotation)

    annotations_text = ''.join(clean_annotations)
    image_tokens = " ".join(["<image>"] * max_images)
    combined_text = f"{image_tokens}{annotations_text}<|endofchunk|>{tokenizer.eos_token}"
    # tokenizer.padding_side="left"
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
    gaze_tensor = gaze_tensor.unsqueeze(1)
    overlaid_tensor = overlaid_tensor.unsqueeze(1)

    del images  # Explicitly delete the list to free up memory
    torch.cuda.empty_cache()  # Use sparingly
    return images_tensor, overlaid_tensor, gaze_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']





def preprocess_gaze_attention(sample, clip_processor, tokenizer, max_images=5, max_tokens=256):
    """
    Decode JSON, base64 images, and process annotations for a single sample.
    """
    # Decode the JSON data for a single sample
    json_data = json.loads(sample['json'].decode('utf-8'))

    images = []
    gaze=[]
    overlaid=[]
    annotations = json_data['annotations']

    # Decode and process each image
    for i in range(0,max_images):
        image_base64 = json_data['images'][i]
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
        processed_image1 = clip_processor(image)  # Convert image to tensor
        heatmap_base64 = json_data['gazemaps'][i]
        heatmap = Image.open(BytesIO(base64.b64decode(heatmap_base64))).convert('RGB')
        processed_image2= clip_processor(heatmap)  # Convert image to tensor
        overlay_base64= json_data['gazemaps'][i]
        overlays = Image.open(BytesIO(base64.b64decode(overlay_base64))).convert('RGB')
        processed_image3= clip_processor(overlays)  # Convert image to tensor
        gaze.append(processed_image2)
        overlaid.append(processed_image3)
        images.append(processed_image1)
        


    if len(images) < max_images:
        raise ValueError(f"Expected {max_images} images, but found {len(images)}")

    # Stack images to create a single tensor
    images_tensor = torch.stack(images)
    gaze_tensor = torch.stack(gaze)
    overlaid_tensor = torch.stack(overlaid)



    clean_annotations = []
    for annotation in annotations:
        # Remove the pattern "number dot space" using regular expressions
        cleaned_annotation = re.sub(r'\d+\.\s', '', annotation)
        clean_annotations.append(cleaned_annotation)

    annotations_text = ''.join(clean_annotations)
    image_tokens = " ".join(["<image>"] * max_images)
    combined_text = f"{image_tokens}{annotations_text}<|endofchunk|>{tokenizer.eos_token}"
    # tokenizer.padding_side="left"
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
    gaze_tensor = gaze_tensor.unsqueeze(1)
    overlaid_tensor = overlaid_tensor.unsqueeze(1)

    del images  # Explicitly delete the list to free up memory
    torch.cuda.empty_cache()  # Use sparingly
    return images_tensor,overlaid_tensor,gaze_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def preprocess_gaze_attention_internvl(sample, clip_processor, tokenizer, max_images=5, max_tokens=128, input_size=448, max_num=1):
    """
    Decode JSON, base64 images, and process annotations for a single sample.
    Incorporates dynamic preprocessing for images.
    """
    # Decode the JSON data for a single sample
    json_data = json.loads(sample['json'].decode('utf-8'))

    images = []
    gaze = []
    overlaid = []
    annotations = json_data['annotations']
    transform = build_transform(input_size=input_size)

    # Decode and process each image
    for i in range(max_images):
        # Decode RGB image
        image_base64 = json_data['overlays'][i]
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
        # Use dynamic preprocessing for image splitting
        processed_images = dynamic_preprocess(image, image_size=input_size, max_num=max_num)

        # Decode gaze heatmap
        # heatmap_base64 = json_data['gazemaps'][i]
        # heatmap = Image.open(BytesIO(base64.b64decode(heatmap_base64))).convert('RGB')
        # #processed_heatmaps = dynamic_preprocess(heatmap, image_size=input_size, max_num=max_num)

        # # Decode gaze-overlayed images
        # overlay_base64 = json_data['overlays'][i]
        # overlay = Image.open(BytesIO(base64.b64decode(overlay_base64))).convert('RGB')
        #processed_overlays = dynamic_preprocess(overlay, image_size=input_size, max_num=max_num)
        overlay_base64= json_data['overlays'][i]
        overlays = Image.open(BytesIO(base64.b64decode(overlay_base64))).convert('RGB')
        processed_image3= clip_processor(overlays)  # Convert image to tensor
        heatmap_base64 = json_data['gazemaps'][i]
        heatmap = Image.open(BytesIO(base64.b64decode(heatmap_base64))).convert('RGB')
        processed_image2= clip_processor(heatmap)  # Convert image to tensor
        gaze.append(processed_image2)
        overlaid.append(processed_image3)
        # Transform all processed images into tensors
        # transform = build_transform(input_size=input_size)
        images.extend([transform(img) for img in processed_images])  # Add all patches for the image
        # gaze.extend([transform(img) for img in processed_heatmaps])
        # overlaid.extend([transform(img) for img in processed_overlays])

        # gaze.append(processed_image2)
        # overlaid.append(processed_image3)

    # Stack image tensors for all patches
    images_tensor = torch.stack(images) 
    gaze_tensor = torch.stack(gaze)
    overlaid_tensor = torch.stack(overlaid)

    # Clean and combine annotations
    clean_annotations = []
    for annotation in annotations:
        # Remove patterns like "number dot space" using regex
        cleaned_annotation = re.sub(r'\d+\.\s', '', annotation)
        clean_annotations.append(cleaned_annotation)

    annotations_text = ''.join(clean_annotations)
    image_tokens = " ".join(["<image>"] * len(images))  # Adjust for dynamic preprocessing
    combined_text = f"{image_tokens}{annotations_text}<|endofchunk|>{tokenizer.eos_token}"

    # Tokenize annotations
    annotations_tokenized = tokenizer(
        combined_text,
        max_length=max_tokens,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    # Reshape tokenized outputs
    input_ids = annotations_tokenized['input_ids'].reshape(-1)  # Flatten
    attention_mask = annotations_tokenized['attention_mask'].reshape(-1)

    images_tensor = images_tensor.unsqueeze(1)
    gaze_tensor = gaze_tensor.unsqueeze(1)
    overlaid_tensor = overlaid_tensor.unsqueeze(1)

    # Free up memory
    del images, gaze, overlaid
    torch.cuda.empty_cache()

    return images_tensor, overlaid_tensor, gaze_tensor, input_ids, attention_mask



# def preprocess_gaze_attention(sample, clip_processor, tokenizer, max_images=5, max_tokens=256):
#     """
#     Decode JSON, base64 images, and process annotations for a single sample.
#     """
#     # Decode the JSON data for a single sample
#     json_data = json.loads(sample['json'].decode('utf-8'))

#     images = []
#     gaze=[]
#     overlaid=[]
#     annotations = json_data['annotations']

#     # Decode and process each image
#     for i in range(0,max_images):
#         image_base64 = json_data['images'][i]
#         image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
#         processed_image1 = clip_processor(image)  # Convert image to tensor

#         overlay_base64= json_data['overlays'][i]
#         overlays = Image.open(BytesIO(base64.b64decode(overlay_base64))).convert('RGB')
#         processed_image3= clip_processor(overlays)  # Convert image to tensor
#         heatmap_base64 = json_data['gazemaps'][i]
#         heatmap = Image.open(BytesIO(base64.b64decode(heatmap_base64))).convert('RGB')
#         processed_image2= clip_processor(heatmap)  # Convert image to tensor
#         gaze.append(processed_image2)
#         overlaid.append(processed_image3)
#         images.append(processed_image1)
        


#     if len(images) < max_images:
#         raise ValueError(f"Expected {max_images} images, but found {len(images)}")

#     # Stack images to create a single tensor
#     images_tensor = torch.stack(images)
#     gaze_tensor = torch.stack(gaze)
#     overlaid_tensor = torch.stack(overlaid)



#     clean_annotations = []
#     for annotation in annotations:
#         # Remove the pattern "number dot space" using regular expressions
#         cleaned_annotation = re.sub(r'\d+\.\s', '', annotation)
#         clean_annotations.append(cleaned_annotation)

#     annotations_text = ''.join(clean_annotations)
#     image_tokens = " ".join(["<image>"] * max_images)
#     # combined_text = f"{image_tokens}{annotations_text}<|endofchunk|>{tokenizer.eos_token}"
#     combined_text = f"{image_tokens}{annotations_text}<|endofchunk|>"
#     # tokenizer.padding_side="left"
#     # annotations_tokenized = tokenizer(
#     #     combined_text,
#     #     max_length=max_tokens,
#     #     truncation=True,
#     #     padding='max_length',
#     #     return_tensors='pt'
#     # )

#     inputs = tokenizer(images= images_tensor.unsqueeze(1),
#             text=combined_text, return_tensors="pt" )
#     labels = tokenizer(
#         text=combined_text,
#         return_tensors="pt" )

#     # annotations_tokenized['input_ids'] = annotations_tokenized['input_ids'].reshape(256)
#     # annotations_tokenized['attention_mask'] = annotations_tokenized['attention_mask'].reshape(256)
#     # images_tensor = images_tensor.unsqueeze(1)
#     # gaze_tensor = gaze_tensor.unsqueeze(1)
#     # overlaid_tensor = overlaid_tensor.unsqueeze(1)

#     # del images  # Explicitly delete the list to free up memory
#     torch.cuda.empty_cache()  # Use sparingly
#     # return images_tensor,overlaid_tensor,gaze_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']
#     return inputs,labels



def get_dataset_gaze_attention(tokenizer, clip_processor, split="train", rank=0, world_size=1,base_path='insert path/datasets/dataset_gaze_caption/'):
    # Define base path and total number of samples per split
    # base_path = 'insert path/datasets/dataset_exp1/'
    num_samples_per_split = {
        "train": 6758,
        "validation": 1690,
        "test": 2112,
    }

    # Define start and end index for each split based on your file naming
    shard_indices = {
        "train": (0, 6757),  # Adjust these values based on your actual file range
        "validation": (0, 1689),  # Adjust these values based on your actual file range
        "test": (0, 2111),   # Adjust these values based on your actual file range
    }


    # num_samples_per_split = {
    #     "train": 6789,
    #     "validation": 1698,
    #     "test": 2122,
    #     # "test": 400,
    # }

    # # Define start and end index for each split based on your file naming
    # shard_indices = {
    #     "train": (0, 6788),  # Adjust these values based on your actual file range
    #     "validation": (0, 1697),  # Adjust these values based on your actual file range
    #     "test": (0, 2121),   # Adjust these values based on your actual file range
    #     #  "test": (0, 399),   # Adjust these values based on your actual file range
    # }
    # num_samples_per_split = {
    #     "train": 200,
    #     "validation": 100,
    #     # "test": 2122,
    #     "test": 100,
    # }

    # # Define start and end index for each split based on your file naming
    # shard_indices = {
    #     "train": (0, 199),  # Adjust these values based on your actual file range
    #     "validation": (0, 99),  # Adjust these values based on your actual file range
    #     # "test": (0, 2121),   # Adjust these values based on your actual file range
    #      "test": (0, 99),   # Adjust these values based on your actual file range
    # }
    # num_samples_per_split = {
    #     "train": 83,
    #     "validation": 21,
    #     "test": 26,
    # }

    # # Define start and end index for each split based on your file naming
    # shard_indices = {
    #     "train": (0, 82),  # Adjust these values based on your actual file range
    #     "validation": (0, 20),  # Adjust these values based on your actual file range
    #     "test": (0, 25),   # Adjust these values based on your actual file range
    # }




    # num_samples_per_split = {
    #     "train": 6851,
    #     "validation": 2141,
    #     "test": 1713,
    # }

    # # Define start and end index for each split based on your file naming
    # shard_indices = {
    #     "train": (0, 6850),  # Adjust these values based on your actual file range
    #     "validation": (0, 2140),  # Adjust these values based on your actual file range
    #     "test": (0, 1712),   # Adjust these values based on your actual file range
    # }

    num_samples = num_samples_per_split[split]
    global_batch_size = 8 * world_size
    num_batches = math.ceil(num_samples / global_batch_size)

    preprocess_fn = functools.partial(preprocess_gaze_attention,
        clip_processor=clip_processor,
        tokenizer=tokenizer)

    # Generate shard list based on the specified range
    shard_list = generate_shard_list(split, base_path, *shard_indices[split])

    # Create the WebDataset
    dataset = wds.WebDataset(shard_list).map(preprocess_fn)
    dataset = dataset.shuffle(num_samples, initial= num_samples)

    # if world_size > 1:
    #     # In distributed mode, ensure data is split correctly across workers
    #     dataset = dataset.shard(num_shards=world_size, shard_id=rank)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=8,
        num_workers=2,  # Adjust according to your setup
        shuffle=False,  # Shuffle only for training data
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=0)


# def get_dataset_gaze_attention(tokenizer, clip_processor, split="train", rank=0, world_size=1,base_path='insert path/datasets/dataset_exp1/'):
#     # Define base path and total number of samples per split
#     # base_path = 'insert path/datasets/dataset_exp1/'
#     num_samples_per_split = {
#         "train": 83,
#         "validation": 21,
#         "test": 26,
#     }

#     # Define start and end index for each split based on your file naming
#     shard_indices = {
#         "train": (0, 82),  # Adjust these values based on your actual file range
#         "validation": (0, 20),  # Adjust these values based on your actual file range
#         "test": (0, 25),   # Adjust these values based on your actual file range
#     }

#     num_samples = num_samples_per_split[split]
#     global_batch_size = 32 * world_size
#     num_batches = math.ceil(num_samples / global_batch_size)

#     preprocess_fn = functools.partial(preprocess_gaze_attention,
#         clip_processor=clip_processor,
#         tokenizer=tokenizer)

#     # Generate shard list based on the specified range
#     shard_list = generate_shard_list(split, base_path, *shard_indices[split])

#     # Create the WebDataset
#     dataset = wds.WebDataset(shard_list).map(preprocess_fn)
#     dataset = dataset.shuffle(num_samples, initial= num_samples)

#     # if world_size > 1:
#     #     # In distributed mode, ensure data is split correctly across workers
#     #     dataset = dataset.shard(num_shards=world_size, shard_id=rank)

#     dataloader = wds.WebLoader(
#         dataset,
#         batch_size=32,
#         num_workers=2,  # Adjust according to your setup
#         shuffle=False,  # Shuffle only for training data
#     )

#     dataloader.num_batches = num_batches
#     dataloader.num_samples = num_samples

#     return DataInfo(dataloader=dataloader, shared_epoch=0)



# def get_dataset_gaze_attention(tokenizer, clip_processor, split="train", rank=0, world_size=1,base_path='insert path/datasets/dataset_exp1/'):
#     # Define base path and total number of samples per split
#     # base_path = 'insert path/datasets/dataset_exp1/'
#     num_samples_per_split = {
#         "train": 6758,
#         "validation": 1690,
#         "test": 2112,
#     }

#     # Define start and end index for each split based on your file naming
#     shard_indices = {
#         "train": (0, 6757),  # Adjust these values based on your actual file range
#         "validation": (0, 1689),  # Adjust these values based on your actual file range
#         "test": (0, 2111),   # Adjust these values based on your actual file range
#     }

#     num_samples = num_samples_per_split[split]
#     global_batch_size = 32 * world_size
#     num_batches = math.ceil(num_samples / global_batch_size)

#     preprocess_fn = functools.partial(preprocess_gaze_attention,
#         clip_processor=clip_processor,
#         tokenizer=tokenizer)

#     # Generate shard list based on the specified range
#     shard_list = generate_shard_list(split, base_path, *shard_indices[split])

#     # Create the WebDataset
#     dataset = wds.WebDataset(shard_list).map(preprocess_fn)
#     dataset = dataset.shuffle(num_samples, initial= num_samples)

#     # if world_size > 1:
#     #     # In distributed mode, ensure data is split correctly across workers
#     #     dataset = dataset.shard(num_shards=world_size, shard_id=rank)

#     dataloader = wds.WebLoader(
#         dataset,
#         batch_size=32,
#         num_workers=2,  # Adjust according to your setup
#         shuffle=False,  # Shuffle only for training data
#     )

#     dataloader.num_batches = num_batches
#     dataloader.num_samples = num_samples

#     return DataInfo(dataloader=dataloader, shared_epoch=0)



def get_dataset_gaze_dist(tokenizer, clip_processor, split="train", rank=0, world_size=1,base_path='insert path/datasets/dataset_exp1/'):
    # Define base path and total number of samples per split
    # base_path = 'insert path/datasets/dataset_exp1/'
    num_samples_per_split = {
        "train": 6758,
        "validation": 1690,
        "test": 2112,
    }

    # Define start and end index for each split based on your file naming
    shard_indices = {
        "train": (0, 6757),  # Adjust these values based on your actual file range
        "validation": (0, 1689),  # Adjust these values based on your actual file range
        "test": (0, 2111),   # Adjust these values based on your actual file range
    }

    num_samples = num_samples_per_split[split]
    global_batch_size = 32 * world_size
    num_batches = math.ceil(num_samples / global_batch_size)

    preprocess_fn = functools.partial(preprocess_gaze_dist,
        clip_processor=clip_processor,
        tokenizer=tokenizer)

    # Generate shard list based on the specified range
    shard_list = generate_shard_list(split, base_path, *shard_indices[split])

    # Create the WebDataset
    dataset = wds.WebDataset(shard_list).map(preprocess_fn)
    dataset = dataset.shuffle(num_samples, initial= num_samples)

    # if world_size > 1:
    #     # In distributed mode, ensure data is split correctly across workers
    #     dataset = dataset.shard(num_shards=world_size, shard_id=rank)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=32,
        num_workers=2,  # Adjust according to your setup
        shuffle=False,  # Shuffle only for training data
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=0)

def preprocess_image(sample, image_processor):
    """
    Convert images to tensors for training.
    Augmentations: random horizontal flip.
    Normalization handled by wds.
    """
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    return image


def filter_no_caption_or_no_image(sample):
    """
    Filter out LAION samples with no caption or no image.
    """
    return ("txt" in sample) and (
        "png" in sample or "jpg" in sample or "jpeg" in sample
    )


def preprocess_laion_text(sample, tokenizer, max_tokens=32):
    """
    Preprocess text for LAION.
    Captions are truncated to 32 tokens by default.
    """
    tokenizer.padding_side = "right"
    sample = [
        (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
    ]
    text = tokenizer(
        sample,
        max_length=max_tokens,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"]


def preprocess_gpt_interleaved(
    info, tokenizer, clip_processor, min_num_images, max_num_images, max_tokens=256
):
    """
    Preprocess a ChatGPT-generated image-text sequence.
    """
    text = info["example"]
    text = re.sub(r"_!_IMAGE\d+_!_", "<|endofchunk|><image>", text)

    # convert images from base64 to PIL
    images = []
    for image_key in range(1, len(info["image_map"]) + 1):
        image_base64 = info["image_map"][f"_!_IMAGE{image_key}_!_"]["base64_image"]
        rawbytes = base64.b64decode(image_base64)
        images.append(Image.open(io.BytesIO(rawbytes)).convert("RGB"))

    # preprocess and pad images
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), max_num_images))
    images_tensors = images_tensors[keep_ixs]
    if len(images_tensors) < max_num_images:
        zero_padding = torch.zeros(
            (max_num_images - len(images_tensors), 3, 224, 224), dtype=torch.float
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # preprocess and tokenize text
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )

    indices = [m.start() for m in re.finditer("<image>", text)]
    if len(indices) > max_num_images:
        start_index = indices[max_num_images - 1]
        text = text[:start_index]

    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # reject sequences with too few images after truncation
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )
    if num_images < min_num_images:
        raise ValueError(f"Fewer than {min_num_images} images in sample")

    return (images_tensors, (text_tensor["input_ids"], text_tensor["attention_mask"]))




#my preprocess_function
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

# Note: You need to ensure that 'clip_processor' is a function that takes a PIL Image as input and returns a processed tensor.
# The 'tokenizer' should be prepared to handle your text appropriately, including setting any necessary special tokens.









def preprocess_interleaved(
    sample,
    tokenizer,
    clip_processor,
    sim_threshold,
    min_num_images,
    max_num_images,
    max_tokens=256,
):
    """
    Preprocess an interleaved image-text sequence, either by calling preprocess_gpt_interleaved (if the sequence
    is ChatGPT-generated) or by preprocessing in this function (if the sequences is from MMC4).
    """
    info = json.loads(sample[0])
    if "is_gpt" in info:
        return preprocess_gpt_interleaved(
            info, tokenizer, clip_processor, min_num_images, max_num_images, max_tokens
        )

    sentences = info["text_list"]
    sim_matrix = info["similarity_matrix"]

    # load images first to find which ones are valid
    valid_images, valid_image_indices = [], []
    for i, sample_image in enumerate(info["image_info"]):
        if "image_base64" not in sample_image:
            continue
        image_base64 = sample_image["image_base64"]
        rawbytes = base64.b64decode(image_base64)

        # filter to images >= 10KB
        if len(rawbytes) // 1000 <= MIN_KB:
            continue

        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
        valid_images.append(image)
        valid_image_indices.append(i)

    if len(valid_image_indices) == 0:
        raise ValueError("No images in sample")

    sim_matrix = np.array(sim_matrix)  # of shape images x sentences
    sim_matrix = sim_matrix[valid_image_indices]

    # negate the similarities to turn then into costs
    cost_matrix = -sim_matrix
    # find one to one assignements
    image_indices, sentence_indices = linear_sum_assignment(cost_matrix)

    images, sentence_ixs = [], []
    for i, sim_ix in zip(image_indices, sentence_indices):
        sim_score = sim_matrix[i][sim_ix]

        if sim_score < sim_threshold:
            continue

        images.append(valid_images[i])
        sentence_ixs.append(sim_ix)

    if len(images) == 0:
        raise ValueError("No images in sample")

    # preprocess and pad images
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), max_num_images))
    images_tensors = images_tensors[keep_ixs]
    sentence_ixs = [sentence_ixs[ix] for ix in keep_ixs]
    if len(images_tensors) < max_num_images:
        zero_padding = torch.zeros(
            (
                max_num_images - len(images_tensors),
                N_CHANNELS,
                images_tensors[0].shape[1],
                images_tensors[0].shape[2],
            ),
            dtype=torch.float,
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # preprocess and tokenize text
    # add in <image> and <eoc> tokens
    for ix in sentence_ixs:
        sentences[ix] = f"<|endofchunk|><image>{sentences[ix]}"
    text = " ".join(sentences)
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )
    if num_images < min_num_images:
        raise ValueError(f"Fewer than {min_num_images} images in sample")
    elif (
        num_images == 1 and random.random() <= 0.5
    ):  # 50% chance of keeping single image samples
        raise ValueError("Only one image in sample")

    # avoid the situation where there's one <image> token and it's at the end
    if (
        num_images == 1
        and text_tensor["input_ids"][:, -1]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    ):
        raise ValueError(
            "Only one image at the end of sample, so labels will all be -100"
        )

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )


def get_mmc4_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    """
    Initialize webdataset for MMC4 / ChatGPT sequences
    """
    input_shards = args.mmc4_shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples_mmc4
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    preprocess_fn = functools.partial(
        preprocess_interleaved,
        clip_processor=image_processor,
        tokenizer=tokenizer,
        sim_threshold=args.mmc4_textsim_threshold,
        min_num_images=args.mmc4_min_num_images,
        max_num_images=args.mmc4_max_num_images,
    )

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.to_tuple("json", handler=log_and_continue),
            wds.map(preprocess_fn, handler=log_and_continue),
            wds.batched(args.batch_size_mmc4, partial=False),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_mmc4 * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


#smaller dataset 
def small_generate_shard_list(split, base_path, start_index, end_index, limit=None):
    """Generate formatted shard paths for a given range, with an optional limit."""
    shard_list = [f"{base_path}/{split}_{i:03d}.tar" for i in range(start_index, end_index + 1)]
    random.Random(4).shuffle(shard_list)  
    if limit:
        shard_list = shard_list[:limit]  # Limit the number of shards used
    return shard_list

def get_small_dataset(tokenizer, clip_processor, split="train", rank=0, world_size=1, sample_limit=None):
    base_path = 'insert path/datasets/modified'
    num_samples_per_split = {
        "train": 6759,
        "validation": 1689,
        "test": 2112,
    }

    shard_indices = {
        "train": (0, 6759),
        "validation": (0, 1689),
        "test": (0, 2112),
    }

    num_samples = num_samples_per_split[split]
    global_batch_size = 32 * world_size
    num_batches = math.ceil(num_samples / global_batch_size)

    preprocess_fn = functools.partial(preprocess,
                                      clip_processor=clip_processor,
                                      tokenizer=tokenizer)

    shard_list = small_generate_shard_list(split, base_path, *shard_indices[split], limit=33)  # Limiting shards

    dataset = wds.WebDataset(shard_list).map(preprocess_fn)
    if sample_limit:
        dataset = dataset.slice(0, sample_limit)  # Taking only a limited number of samples

    # Optional: Shuffle and distributed settings
    # dataset = dataset.shuffle(num_samples, initial=num_samples)
    # if world_size > 1:
    #     dataset = dataset.shard(num_shards=world_size, shard_id=rank)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        shuffle=False,
    )

    # Adjust the number of batches and samples to match the sample limit
    dataloader.num_batches = math.ceil(sample_limit / 32) if sample_limit else num_batches
    dataloader.num_samples = sample_limit if sample_limit else num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=0)



def get_small_dataset_gaze(tokenizer, clip_processor, split="train", rank=0, world_size=1, sample_limit=None):
    base_path = 'insert path/datasets/dataset_w_gaze/dataset_wgaze/modified'
    num_samples_per_split = {
        "train": 6759,
        "validation": 1689,
        "test": 2112,
    }

    shard_indices = {
        "train": (0, 6759),
        "validation": (0, 1689),
        "test": (0, 2112),
    }

    num_samples = num_samples_per_split[split]
    global_batch_size = 32 * world_size
    num_batches = math.ceil(num_samples / global_batch_size)

    preprocess_fn = functools.partial(preprocess_gaze,
                                      clip_processor=clip_processor,
                                      tokenizer=tokenizer)

    shard_list = small_generate_shard_list(split, base_path, *shard_indices[split], limit=33)  # Limiting shards

    dataset = wds.WebDataset(shard_list).map(preprocess_fn)
    if sample_limit:
        dataset = dataset.slice(0, sample_limit)  # Taking only a limited number of samples

    # Optional: Shuffle and distributed settings
    # dataset = dataset.shuffle(num_samples, initial=num_samples)
    # if world_size > 1:
    #     dataset = dataset.shard(num_shards=world_size, shard_id=rank)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        shuffle=False,
    )

    # Adjust the number of batches and samples to match the sample limit
    dataloader.num_batches = math.ceil(sample_limit / 32) if sample_limit else num_batches
    dataloader.num_samples = sample_limit if sample_limit else num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=0)


#newest part rn
def generate_shard_list(split, base_path, start_index, end_index):
    """Generate formatted shard paths for a given range."""
    # Adjust format based on how your files are named (e.g., 'train_000.tar')
    shard_list = [f"{base_path}/{split}_{i:03d}.tar" for i in range(start_index, end_index + 1)]
    return shard_list

# def get_dataset_new(tokenizer, clip_processor, split="train", rank=0, world_size=1):
#     # Define base path and total number of samples per split
#     base_path = '/home/pani3/dataset_obs_less/'
#     num_samples_per_split = {
#         "train": 6789,
#         "validation": 1698,
#         "test": 2122,
#     }

#     # Define start and end index for each split based on your file naming
#     shard_indices = {
#         "train": (0, 6788),  # Adjust these values based on your actual file range
#         "validation": (0, 1697),  # Adjust these values based on your actual file range
#         "test": (0, 2121),   # Adjust these values based on your actual file range
#     }

#     num_samples = num_samples_per_split[split]
#     global_batch_size = 32 * world_size
#     num_batches = math.ceil(num_samples / global_batch_size)

#     preprocess_fn = functools.partial(preprocess,
#         clip_processor=clip_processor,
#         tokenizer=tokenizer)

#     # Generate shard list based on the specified range
#     shard_list = generate_shard_list(split, base_path, *shard_indices[split])

#     # Create the WebDataset
#     dataset = wds.WebDataset(shard_list).map(preprocess_fn)
#     dataset = dataset.shuffle(num_samples, initial= num_samples)

#     # if world_size > 1:
#     #     # In distributed mode, ensure data is split correctly across workers
#     #     dataset = dataset.shard(num_shards=world_size, shard_id=rank)

#     dataloader = wds.WebLoader(
#         dataset,
#         batch_size=32,
#         num_workers=2,  # Adjust according to your setup
#         shuffle=False,  # Shuffle only for training data
#     )

#     dataloader.num_batches = num_batches
#     dataloader.num_samples = num_samples

#     return DataInfo(dataloader=dataloader, shared_epoch=0)


def preprocess2(sample, clip_processor, tokenizer, max_gaze_points=6, max_images=5,max_tokens=256):
    """
    Decode JSON, process gaze points as text prompts, and handle annotations for a single sample.
    """
    # Decode the JSON data for a single sample
    json_data = json.loads(sample['json'].decode('utf-8'))

    # Extract gaze points
    gaze_points = json_data['gazepoints']

    images = []
    annotations = json_data['annotations']

    # Decode and process each image
    for i in range(0,max_images):
        image_base64 = json_data['images'][i]
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
        processed_image = clip_processor(image)  # Convert image to tensor
        images.append(processed_image)    

    if len(images) < max_images:
        raise ValueError(f"Expected {max_images} images, but found {len(images)}")

    # Stack images to create a single tensor
    images_tensor = torch.stack(images)

    # if len(gaze_points) < max_gaze_points:
    #     raise ValueError(f"Expected {max_gaze_points} gaze points, but found {len(gaze_points)}")

    # Convert gaze points to textual representation (e.g., "gaze at (x1, y1), (x2, y2), ...")
    gaze_points_size= len(gaze_points)
    gaze_text = " ".join([f"gaze at ({int(gaze[0][0])}, {int(gaze[0][1])})" for gaze in gaze_points[:gaze_points_size]])

    # Extract and clean the annotations
    annotations = json_data['annotations']
    clean_annotations = []
    for annotation in annotations:
        # Remove the pattern "number dot space" using regular expressions
        cleaned_annotation = re.sub(r'\d+\.\s', '', annotation)
        clean_annotations.append(cleaned_annotation)


    image_tokens = " ".join(["<image>"] * max_images)

    # Concatenate annotations and gaze points into a single text prompt
    annotations_text = ' '.join(clean_annotations)
    combined_text = f"{image_tokens} and these are the gaze points {gaze_text}. {annotations_text}<|endofchunk|>{tokenizer.eos_token}"

    # Tokenize the combined text
    annotations_tokenized = tokenizer(
        combined_text,
        max_length=max_tokens,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    # Reshape the tokenized input to the expected format
    annotations_tokenized['input_ids'] = annotations_tokenized['input_ids'].reshape(max_tokens)
    annotations_tokenized['attention_mask'] = annotations_tokenized['attention_mask'].reshape(max_tokens)
    images_tensor = images_tensor.unsqueeze(1)

    del images  # Explicitly delete the list to free up memory
    torch.cuda.empty_cache()  # Use sparingly
    return images_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']


def preprocess3(sample, clip_processor, tokenizer, max_gaze_points=6, max_images=5, max_tokens=256):
    """
    Decode JSON, process gaze points as text prompts, and handle annotations for a single sample.
    """
    # Decode the JSON data for a single sample
    json_data = json.loads(sample['json'].decode('utf-8'))
    input_size=448
    max_tokens=256

    # Extract gaze points
    gaze_points = json_data['gazepoints']

    images = []
    annotations = json_data['annotations']
    transform = build_transform(input_size=input_size)

    # Decode and process each image
    for i in range(max_images):
        image_base64 = json_data['images'][i]
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
        # Use dynamic preprocessing for image splitting
        processed_images = dynamic_preprocess(image, image_size=input_size, max_num=1)
        images.extend([transform(img) for img in processed_images]) 
        # image_base64 = json_data['images'][i]
        # image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
        # processed_image = clip_processor(image)  # Convert image to tensor
        # images.append(processed_image)

    if len(images) < max_images:
        raise ValueError(f"Expected {max_images} images, but found {len(images)}")

    # Stack images to create a single tensor
    images_tensor = torch.stack(images)

    # Process gaze points into text format (up to max_gaze_points)
    gaze_text = " ".join([f"gaze at ({int(gaze[0][0])}, {int(gaze[0][1])})" for gaze in gaze_points[:max_gaze_points]])

    # Extract and clean the annotations
    clean_annotations = []
    for annotation in annotations:
        # Remove the pattern "number dot space" using regular expressions
        cleaned_annotation = re.sub(r'\d+\.\s', '', annotation)
        clean_annotations.append(cleaned_annotation)

    # Add <image> tokens for each image
    image_tokens = " ".join(["<image>"] * max_images)

    # Concatenate annotations, image tokens, and gaze points into a single text prompt
    annotations_text = ' '.join(clean_annotations)
    combined_text = f"{image_tokens} {gaze_text}. {annotations_text} <|endofchunk|>{tokenizer.eos_token}"

    # Tokenize the combined text and request token offsets
    annotations_tokenized = tokenizer(
        combined_text,
        max_length=max_tokens,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        #return_offsets_mapping=True  # Return character-to-token mapping
    )


    # Reshape the tokenized input and attention mask to the expected format

    annotations_tokenized['input_ids'] = annotations_tokenized['input_ids'].reshape(256)
    annotations_tokenized['attention_mask'] = annotations_tokenized['attention_mask'].reshape(256)

    # Reshape the images tensor
    images_tensor = images_tensor.unsqueeze(1)

    del images  # Explicitly delete the list to free up memory
    torch.cuda.empty_cache()  # Use sparingly to clear memory on the GPU

    return images_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']


def preprocess(sample, clip_processor, tokenizer, max_images=1, max_tokens=256):
    """
    Decode JSON, base64 images, and process annotations for a single sample.
    """
    # Decode the JSON data for a single sample
    json_data = json.loads(sample['json'].decode('utf-8'))

    images = []
    annotations = json_data['annotations']

    # Decode and process each image
    for i in range(0,max_images):
        image_base64 = json_data['images'][i]
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
        processed_image = clip_processor(image)  # Convert image to tensor
        images.append(processed_image)

    if len(images) < max_images:
        raise ValueError(f"Expected {max_images} images, but found {len(images)}")

    # Stack images to create a single tensor
    images_tensor = torch.stack(images)

    clean_annotations = []
    for annotation in annotations:
        # Remove the pattern "number dot space" using regular expressions
        cleaned_annotation = re.sub(r'\d+\.\s', '', annotation)
        clean_annotations.append(cleaned_annotation)

    annotations_text = ' '.join(clean_annotations)


    # Concatenate annotations and tokenize
    # annotations_text = ' '.join(annotations)
    image_tokens = " ".join(["<image>"] * max_images)
    combined_text = f"{image_tokens}{annotations_text}<|endofchunk|>{tokenizer.eos_token}"

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

    del images  # Explicitly delete the list to free up memory
    torch.cuda.empty_cache()  # Use sparingly
    return images_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']


# def preprocess(sample, clip_processor, tokenizer, max_images=1, max_tokens=256):
#     """
#     Decode JSON, base64 images, and process annotations for a single sample.
#     """
#     # Decode the JSON data for a single sample
#     json_data = json.loads(sample['json'].decode('utf-8'))

#     images = []
#     annotations = json_data['annotations']

#     # Decode and process each image
#     for i in range(0,max_images):
#         image_base64 = json_data['overlays'][i]
#         image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
#         processed_image = clip_processor(image)  # Convert image to tensor
#         images.append(processed_image)

#     if len(images) < max_images:
#         raise ValueError(f"Expected {max_images} images, but found {len(images)}")

#     # Stack images to create a single tensor
#     images_tensor = torch.stack(images)

#     clean_annotations = []
#     for annotation in annotations:
#         # Remove the pattern "number dot space" using regular expressions
#         cleaned_annotation = re.sub(r'\d+\.\s', '', annotation)
#         clean_annotations.append(cleaned_annotation)

#     annotations_text = ' '.join(clean_annotations)


#     # Concatenate annotations and tokenize
#     # annotations_text = ' '.join(annotations)
#     image_tokens = " ".join(["<image>"] * max_images)
#     combined_text = f"{image_tokens}{annotations_text}<|endofchunk|>{tokenizer.eos_token}"

#     annotations_tokenized = tokenizer(
#         combined_text,
#         max_length=max_tokens,
#         truncation=True,
#         padding='max_length',
#         return_tensors='pt'
#     )

#     annotations_tokenized['input_ids'] = annotations_tokenized['input_ids'].reshape(256)
#     annotations_tokenized['attention_mask'] = annotations_tokenized['attention_mask'].reshape(256)
#     images_tensor = images_tensor.unsqueeze(1)

#     del images  # Explicitly delete the list to free up memory
#     torch.cuda.empty_cache()  # Use sparingly
#     return images_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']



def get_dataset_text(tokenizer, clip_processor, split="train", rank=0, world_size=1):
    # Define base path and total number of samples per split
    base_path = 'insert path/datasets/dataset_gaze_text/'
    # num_samples_per_split = {
    #     "train": 6758,
    #     "validation": 1689,
    #     "test": 2112,
    # }

    # # Define start and end index for each split based on your file naming
    # shard_indices = {
    #     "train": (0, 6757),  # Adjust these values based on your actual file range
    #     "validation": (0, 1688),  # Adjust these values based on your actual file range
    #     "test": (0, 2111),   # Adjust these values based on your actual file range
    # }
    num_samples_per_split = {
        "train": 200,
        "validation": 100,
        # "test": 2122,
        "test": 100,
    }

    # Define start and end index for each split based on your file naming
    shard_indices = {
        "train": (0, 199),  # Adjust these values based on your actual file range
        "validation": (0, 99),  # Adjust these values based on your actual file range
        # "test": (0, 2121),   # Adjust these values based on your actual file range
         "test": (0, 99),   # Adjust these values based on your actual file range
    }
    num_samples = num_samples_per_split[split]
    global_batch_size = 8 * world_size
    num_batches = math.ceil(num_samples / global_batch_size)

    preprocess_fn = functools.partial(preprocess3,
        clip_processor=clip_processor,
        tokenizer=tokenizer)

    # Generate shard list based on the specified range
    shard_list = generate_shard_list(split, base_path, *shard_indices[split])

    # Create the WebDataset
    dataset = wds.WebDataset(shard_list).map(preprocess_fn)
    dataset = dataset.shuffle(num_samples, initial= num_samples)

    # if world_size > 1:
    #     # In distributed mode, ensure data is split correctly across workers
    #     dataset = dataset.shard(num_shards=world_size, shard_id=rank)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=8,
        num_workers=2,  # Adjust according to your setup
        shuffle=False,  # Shuffle only for training data
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=0)




def get_dataset_new(tokenizer, clip_processor, split="train", rank=0, world_size=1):
    # Define base path and total number of samples per split
    base_path = 'insert path/datasets/dataset_gaze_caption'
    num_samples_per_split = {
        "train": 6851,
        "validation": 2141,
        "test": 1713,
    }

    # Define start and end index for each split based on your file naming
    shard_indices = {
        "train": (0, 6850),  # Adjust these values based on your actual file range
        "validation": (0, 2140),  # Adjust these values based on your actual file range
        "test": (0, 1712),   # Adjust these values based on your actual file range
    }
    # num_samples_per_split = {
    #     "train": 6758,
    #     "validation": 1689,
    #     "test": 2112,
    # }

    # # Define start and end index for each split based on your file naming
    # shard_indices = {
    #     "train": (0, 6757),  # Adjust these values based on your actual file range
    #     "validation": (0, 1688),  # Adjust these values based on your actual file range
    #     "test": (0, 2111),   # Adjust these values based on your actual file range
    # }
    # num_samples_per_split = {
    #     "train": 83,
    #     "validation": 21,
    #     "test": 26,
    # }

    # # Define start and end index for each split based on your file naming
    # shard_indices = {
    #     "train": (0, 82),  # Adjust these values based on your actual file range
    #     "validation": (0, 20),  # Adjust these values based on your actual file range
    #     "test": (0, 25),   # Adjust these values based on your actual file range
    # }
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
    dataset = dataset.shuffle(num_samples, initial= num_samples)

    # if world_size > 1:
    #     # In distributed mode, ensure data is split correctly across workers
    #     dataset = dataset.shard(num_shards=world_size, shard_id=rank)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=32,
        num_workers=2,  # Adjust according to your setup
        shuffle=False,  # Shuffle only for training data
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=0)


def preprocess_gaze(sample, clip_processor, tokenizer, max_images=5, max_tokens=512):
    """
    Decode JSON, base64 images, and process annotations for a single sample.
    """
    # Decode the JSON data for a single sample
    json_data = json.loads(sample['json'].decode('utf-8'))

    images = []
    annotations = json_data['annotations']

    # Decode and process each image
    for i in range(0,max_images):
        image_base64 = json_data['images'][i]
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
        processed_image1 = clip_processor(image)  # Convert image to tensor
        heatmap_base64 = json_data['heatmaps'][i]
        heatmap = Image.open(BytesIO(base64.b64decode(heatmap_base64))).convert('RGB')
        processed_image2= clip_processor(heatmap)  # Convert image to tensor
        images.append(processed_image2)
        images.append(processed_image1)


    if len(images) < max_images:
        raise ValueError(f"Expected {max_images} images, but found {len(images)}")

    # Stack images to create a single tensor
    images_tensor = torch.stack(images)


    clean_annotations = []
    for annotation in annotations:
        # Remove the pattern "number dot space" using regular expressions
        cleaned_annotation = re.sub(r'\d+\.\s', '', annotation)
        clean_annotations.append(cleaned_annotation)

    annotations_text = 'What will happen after the sequence of images just shown? '.join(clean_annotations)

    # Concatenate annotations and tokenize
    #annotations_text = ' '.join(annotations)
    image_tokens = " ".join(["<image>"] * max_images)
    combined_text = f"{image_tokens}{annotations_text}<|endofchunk|>{tokenizer.eos_token}"
    # tokenizer.padding_side="left"
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

    del images  # Explicitly delete the list to free up memory
    torch.cuda.empty_cache()  # Use sparingly
    return images_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']


def preprocess_eval_gaze(sample, clip_processor, tokenizer, max_images=5, max_tokens=256):
    """
    Decode JSON, base64 images, and process annotations for a single sample.
    """
    # Decode the JSON data for a single sample
    json_data = json.loads(sample['json'].decode('utf-8'))

    images = []
    annotations = json_data['annotations']

    # Decode and process each image
    for i in range(0,max_images):
        image_base64 = json_data['images'][i]
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
        processed_image1 = clip_processor(image)  # Convert image to tensor
        heatmap_base64 = json_data['heatmaps'][i]
        heatmap = Image.open(BytesIO(base64.b64decode(heatmap_base64))).convert('RGB')
        processed_image2= clip_processor(heatmap)  # Convert image to tensor
        images.append(processed_image1)
        images.append(processed_image2)


    if len(images) < max_images:
        raise ValueError(f"Expected {max_images} images, but found {len(images)}")

    # Stack images to create a single tensor
    images_tensor = torch.stack(images)




    # Concatenate annotations and tokenize
    annotations_text = ' '.join(annotations)
    image_tokens = " ".join(["<image><image>"] * max_images)
    combined_text = f"{image_tokens}{annotations_text}<|endofchunk|>{tokenizer.eos_token}"
    # tokenizer.padding_side="left"
    annotations_tokenized = tokenizer(
        combined_text,
        max_length=max_tokens,
        truncation=False,
        padding='max_length',
        return_tensors='pt'
    )

    annotations_tokenized['input_ids'] = annotations_tokenized['input_ids'].reshape(256)
    annotations_tokenized['attention_mask'] = annotations_tokenized['attention_mask'].reshape(256)
    images_tensor = images_tensor.unsqueeze(1)

    del images  # Explicitly delete the list to free up memory
    torch.cuda.empty_cache()  # Use sparingly
    return images_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']






def preprocess_eval_base(sample, clip_processor, tokenizer, max_images=5, max_tokens=256):
    """
    Decode JSON, base64 images, and process annotations for a single sample.
    """
    # Decode the JSON data for a single sample
    json_data = json.loads(sample['json'].decode('utf-8'))

    images = []
    annotations = json_data['annotations']

    # Decode and process each image
    for i in range(0,max_images):
        image_base64 = json_data['images'][i]
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
        processed_image1 = clip_processor(image)  # Convert image to tensor
        # heatmap_base64 = json_data['heatmaps'][i]
        # heatmap = Image.open(BytesIO(base64.b64decode(heatmap_base64))).convert('RGB')
        # processed_image2= clip_processor(heatmap)  # Convert image to tensor
        images.append(processed_image1)
        # images.append(processed_image2)


    if len(images) < max_images:
        raise ValueError(f"Expected {max_images} images, but found {len(images)}")

    # Stack images to create a single tensor
    images_tensor = torch.stack(images)




    # Concatenate annotations and tokenize
    annotations_text = ' '.join(annotations)
    image_tokens = " ".join(["<image>"] * max_images)
    combined_text = f"{image_tokens}{annotations_text}<|endofchunk|>{tokenizer.eos_token}"
    # tokenizer.padding_side="left"
    annotations_tokenized = tokenizer(
        combined_text,
        max_length=max_tokens,
        truncation=False,
        padding='max_length',
        return_tensors='pt'
    )

    annotations_tokenized['input_ids'] = annotations_tokenized['input_ids'].reshape(256)
    annotations_tokenized['attention_mask'] = annotations_tokenized['attention_mask'].reshape(256)
    images_tensor = images_tensor.unsqueeze(1)

    del images  # Explicitly delete the list to free up memory
    torch.cuda.empty_cache()  # Use sparingly
    return images_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']



def get_dataset_gaze(tokenizer, clip_processor, split="train", rank=0, world_size=1):
    # Define base path and total number of samples per split
    base_path = 'insert path/datasets/dataset_w_gaze/dataset_wgaze/modified'
    num_samples_per_split = {
        "train": 6759,
        "validation": 1689,
        "test": 2112,
    }

    # Define start and end index for each split based on your file naming
    shard_indices = {
        "train": (0, 6759),  # Adjust these values based on your actual file range
        "validation": (0, 1689),  # Adjust these values based on your actual file range
        "test": (0, 2112),   # Adjust these values based on your actual file range
    }

    num_samples = num_samples_per_split[split]
    global_batch_size = 32 * world_size
    num_batches = math.ceil(num_samples / global_batch_size)

    preprocess_fn = functools.partial(preprocess_gaze,
        clip_processor=clip_processor,
        tokenizer=tokenizer)

    # Generate shard list based on the specified range
    shard_list = generate_shard_list(split, base_path, *shard_indices[split])

    # Create the WebDataset
    dataset = wds.WebDataset(shard_list).map(preprocess_fn)
    dataset = dataset.shuffle(num_samples, initial= num_samples)

    # if world_size > 1:
    #     # In distributed mode, ensure data is split correctly across workers
    #     dataset = dataset.shard(num_shards=world_size, shard_id=rank)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=32,
        num_workers=4,  # Adjust according to your setup
        shuffle=False,  # Shuffle only for training data
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=0)



def get_dataset_gaze_eval(tokenizer, clip_processor, split="train", rank=0, world_size=1):
    # Define base path and total number of samples per split
    base_path = 'insert path/datasets/dataset_w_gaze/dataset_wgaze/modified'
    num_samples_per_split = {
        "train": 32,
        "validation": 32,
        "test": 32,
    }
    # Define start and end index for each split based on your file naming
    shard_indices = {
        "train": (0, 31),  # Adjust these values based on your actual file range
        "validation": (0, 31),  # Adjust these values based on your actual file range
        "test": (0, 31),   # Adjust these values based on your actual file range
    }

    num_samples = num_samples_per_split[split]
    global_batch_size = 32 * world_size
    num_batches = math.ceil(num_samples / global_batch_size)

    preprocess_fn = functools.partial(preprocess_eval_gaze,
        clip_processor=clip_processor,
        tokenizer=tokenizer)

    # Generate shard list based on the specified range
    shard_list = generate_shard_list(split, base_path, *shard_indices[split])

    # Create the WebDataset
    dataset = wds.WebDataset(shard_list).map(preprocess_fn)
    dataset = dataset.shuffle(num_samples, initial= num_samples)

    # if world_size > 1:
    #     # In distributed mode, ensure data is split correctly across workers
    #     dataset = dataset.shard(num_shards=world_size, shard_id=rank)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=32,
        num_workers=1,  # Adjust according to your setup
        shuffle=False,  # Shuffle only for training data
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=0)






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
        preprocess_interleaved_custom,  # This needs to be defined by you.
        clip_processor=image_processor,
        tokenizer=tokenizer,
    )
    # pipeline = [wds.SimpleShardList(input_shards)]

    # Define your data pipeline
    pipeline = [
        wds.SimpleShardList(input_shards),
        # wds.WebDataset(input_shards),
        #wds.WebDaSimpleShardListtaset(input_shards),
        wds.shuffle(10000, initial=10000),
        # wds.decode(),  # Decodes images using PIL; adjust as needed.
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








def get_laion_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    """
    Initialize webdataset for LAION data
    """
    input_shards = args.laion_shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples_laion
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # create two preprocess functions that take in the passed in image_processor and tokenizer
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_laion_text, tokenizer=tokenizer)

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.to_tuple("jpg;png;jpeg", "txt", handler=log_and_continue),
            wds.batched(args.batch_size_laion, partial=False),
            wds.map_tuple(
                preprocess_image_fn, preprocess_text_fn, handler=log_and_continue
            ),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_laion * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_dataset_fn(dataset_type):
    """
    Helper function to get the dataset function based on the dataset type
    """
    if dataset_type == "image_text":
        return get_laion_dataset
    elif dataset_type == "mmc4":
        return get_mmc4_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, image_processor, tokenizer, dataset_type, epoch=0):
    """
    Interface for getting the webdatasets
    """
    return get_dataset_fn(dataset_type)(
        args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer
    )




# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
# bash miniconda3/miniconda.sh -b -u -p miniconda3
# rm -rf miniconda3/miniconda.sh
# miniconda3/bin/conda init bash
# miniconda3/bin/conda init zsh