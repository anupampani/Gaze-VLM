
import json
import torch
import base64
from PIL import Image
import io
import numpy as np
from scipy.optimize import linear_sum_assignment

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
    info = json.loads(sample[0])  # Loading the sample, which includes images and annotations

    # Initialize lists to hold processed images and their indices
    images = []
    for i, image_info in enumerate(info['image_info'][:max_num_images]):
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



#preprocess_new 
import json
import base64
from PIL import Image
from io import BytesIO
import torch

def preprocess(sample, clip_processor, tokenizer, max_images=5, max_tokens=256):
    """
    Decode JSON, base64 images, and process annotations for a single sample.
    """
    # Decode the JSON data for a single sample
    json_data = json.loads(sample['json'].decode('utf-8'))

    images = []
    annotations = json_data['annotations']

    # Decode and process each image
    for img_data in json_data['images'][:max_images]:
        image_base64 = img_data['image_base64']
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
        processed_image = clip_processor(image)  # Convert image to tensor
        images.append(processed_image)

    if len(images) < max_images:
        raise ValueError(f"Expected {max_images} images, but found {len(images)}")

    # Stack images to create a single tensor
    images_tensor = torch.stack(images)

    # Concatenate annotations and tokenize
    annotations_text = ' '.join(annotations)
    annotations_tokenized = tokenizer(
        annotations_text,
        max_length=max_tokens,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    return images_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']






#NEW ONE WHICH WORKS
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
    annotations_tokenized = tokenizer(
        combined_text,
        max_length=max_tokens,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    return images_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']