# def get_dataset1(tokenizer,clip_processor,split="train"):
#     """
#     Initialize a webdataset for the custom dataset.
#     """
#     # Define the path to the shards based on the split
#     input_shards = '/mnt/lv1/ego4d/v2/gazevlm_data/dataset_new/train.tar'
#     assert input_shards is not None, f"Path to {split} shards must be specified."


#     num_samples = 67593
#     global_batch_size = 32*4
#     num_batches = math.ceil(num_samples / global_batch_size)
    

#     # Use functools.partial to prepare the actual preprocess function with any required arguments
#     preprocess_fn = functools.partial(preprocess,  # This needs to be defined by you.
#         clip_processor=clip_processor,
#         tokenizer=tokenizer,)

#     # Initialize the WebDataset
#     dataset = wds.WebDataset(input_shards)
    
#     # Apply preprocessing to each item
#     dataset = dataset.map(preprocess_fn)

#     # Create a DataLoader
#     dataloader = wds.WebLoader(
#         dataset,
#         batch_size=32,  # Ensure you have args.batch_size defined
#         num_workers=4,    # Ensure you have args.workers defined
#         shuffle=(split=="test")     # Shuffle only training data
#     )

#     dataloader.num_batches = num_batches
#     dataloader.num_samples = num_samples
#     return DataInfo(dataloader=dataloader, shared_epoch=0)


#new update made
def get_dataset1(tokenizer, clip_processor, split="train", rank=None, world_size=None):
    # Define the path to the shards based on the split
    input_shards = '/mnt/lv1/ego4d/v2/gazevlm_data/dataset_new/train.tar'
    assert input_shards is not None, f"Path to {split} shards must be specified."

    num_samples = 67593  # Adjust as per your dataset
    global_batch_size = 32 * 4
    num_batches = math.ceil(num_samples / global_batch_size)

    preprocess_fn = functools.partial(preprocess,
        clip_processor=clip_processor,
        tokenizer=tokenizer)

    dataset = wds.WebDataset(input_shards)
    dataset = dataset.map(preprocess_fn)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        shuffle=(split == "train")  # Assuming you want to shuffle only the training data
    )

    # Assuming you have num_batches and num_samples calculated correctly
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=0)


def preprocess(sample, clip_processor, tokenizer, max_images=5, max_tokens=256):
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

    # Concatenate annotations and tokenize
    annotations_text = ' '.join(annotations)
    annotations_tokenized = tokenizer(
        annotations_text,
        max_length=max_tokens,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    annotations_tokenized['input_ids'] = annotations_tokenized['input_ids'].reshape(256)
    annotations_tokenized['attention_mask'] = annotations_tokenized['attention_mask'].reshape(256)
    images_tensor = images_tensor.unsqueeze(1)

    return images_tensor, annotations_tokenized['input_ids'], annotations_tokenized['attention_mask']