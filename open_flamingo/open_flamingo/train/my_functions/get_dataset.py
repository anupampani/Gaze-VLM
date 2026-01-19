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
        input_shards = args.train_shards
    elif split == 'val':
        input_shards = args.val_shards
    elif split == 'test':
        input_shards = args.test_shards
    else:
        raise ValueError(f"Invalid dataset split: {split}")
    
    assert input_shards is not None, f"Input shards for {split} not specified."

    preprocess_fn = functools.partial(
        preprocess_custom,  # This needs to be defined by you.
        image_processor=image_processor,
        tokenizer=tokenizer,
    )

    # Define your data pipeline
    pipeline = [
        wds.WebDataset(input_shards),
        wds.shuffle(10000, initial=10000),
        wds.decode("pil"),  # Decodes images using PIL; adjust as needed.
        wds.to_tuple("jpg;png", "json"),  # Adjust based on your data format.
        wds.map(preprocess_fn),
        wds.batched(args.batch_size, partial=False),
    ]

    dataset = wds.DataPipeline(*pipeline)

    # Assuming num_samples is predefined or calculated elsewhere (replace it with manual numbers)
    num_samples = getattr(args, f"{split}_num_samples")
    global_batch_size = args.batch_size * args.world_size
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

    return dataloader


#New one 
def get_dataset1(args,tokenizer,clip_processor,split="test"):
    """
    Initialize a webdataset for the custom dataset.
    """
    # Define the path to the shards based on the split
    input_shards = '/mnt/lv1/ego4d/v2/gazevlm_data/dataset_new/test.tar'
    assert input_shards is not None, f"Path to {split} shards must be specified."


    num_samples = 67593
    global_batch_size = 32*4
    num_batches = math.ceil(num_samples / global_batch_size)
    

    # Use functools.partial to prepare the actual preprocess function with any required arguments
    preprocess_fn = functools.partial(preprocess,  # This needs to be defined by you.
        clip_processor=clip_processor,
        tokenizer=tokenizer,)

    # Initialize the WebDataset
    dataset = wds.WebDataset(input_shards)
    
    # Apply preprocessing to each item
    dataset = dataset.map(preprocess_fn)

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,  # Ensure you have args.batch_size defined
        num_workers=4,    # Ensure you have args.workers defined
        shuffle=(split=="train")     # Shuffle only training data
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    print('sample done ')

    return DataInfo(dataloader=dataloader, shared_epoch=0)
