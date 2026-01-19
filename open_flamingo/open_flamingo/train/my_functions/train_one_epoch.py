#my train_one_epoch
def train_one_epoch_new(
    args,
    model,
    epoch,
    custom_loader,  # Your custom dataset loader
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    # Setup loader
    num_batches_per_epoch = custom_loader.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs
    autocast = get_autocast(args.precision, cache_enabled=(not args.fsdp))
    cast_dtype = get_cast_dtype(args.precision)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device3= torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # Setup model
    model.train()
    # Setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()


    for num_steps, batch in enumerate(custom_loader):
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        # Forward pass for your custom dataset
        

        with autocast():
            images, input_ids, attention_mask = (
                batch[0].to(device_id, dtype=cast_dtype, non_blocking=True),
                batch[1].to(device_id, dtype=cast_dtype, non_blocking=True),
                batch[2].to(device_id, dtype=cast_dtype, non_blocking=True),
            )

            # Labels setup: shifted inside the model or beforehand
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            labels = labels.to(device_id)

            loss = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )[0] / args.gradient_accumulation_steps

            loss.backward()

            # Gradient accumulation and optimizer step
            if num_steps % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Logging
                if args.rank == 0:
                    wandb.log({
                        "loss": loss.item(),
                        "global_step": global_step,
                    })

        # Log loss to console
        if args.rank == 0 and (num_steps % args.logging_steps == 0):
            print(f"Epoch [{epoch+1}/{args.num_epochs}], "
                  f"Step [{num_steps}/{num_batches_per_epoch}], "
                  f"Loss: {loss.item():.4f}")

        step_time_m.update(time.time() - end)
        end = time.time()

    # Final logging for the epoch
    if args.rank == 0:
        print(f"Epoch {epoch+1} completed. Final Loss: {loss.item():.4f}")






def train_one_epoch(
    args,
    model,
    epoch,
    laion_loader,
    mmc4_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    # setup loaders
    num_batches_per_epoch_laion = laion_loader.num_batches
    num_batches_per_epoch_mmc4 = mmc4_loader.num_batches
    assert (
        num_batches_per_epoch_laion == num_batches_per_epoch_mmc4
    ), "Number of batches in laion and mmc4 datasets must be the same"
    num_batches_per_epoch = num_batches_per_epoch_mmc4
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )  # if fsdp, disable cache to save memory
    cast_dtype = get_cast_dtype(args.precision)

    # setup model
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]
    model.train()

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # loop through dataloader
    for num_steps, (batch_laion, batch_mmc4) in tqdm(
        enumerate(zip(laion_loader, mmc4_loader)),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        #### LAION FORWARD PASS ####
        images = batch_laion[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)
        input_ids = batch_laion[1][0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch_laion[1][1].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            loss_laion = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

        divided_loss_laion = loss_laion / args.gradient_accumulation_steps
        (divided_loss_laion * args.loss_multiplier_laion).backward()

        #### MMC4 FORWARD PASS ####
        images = batch_mmc4[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "b (t f) c h w -> b t f c h w", f=1)
        input_ids = torch.stack([x[0] for x in batch_mmc4[1]]).squeeze(1)
        attention_mask = torch.stack([x[1] for x in batch_mmc4[1]]).squeeze(1)

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        for i in range(labels.shape[0]):
            # remove loss for any token before the first <image> token
            label_idx = 0
            while (
                label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id
            ):
                labels[i][label_idx] = -100
                label_idx += 1

            # get index of all endofchunk tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while (
                    token_idx < labels.shape[1]
                    and labels[i][token_idx] != media_token_id
                ):
                    labels[i][token_idx] = -100
                    token_idx += 1

        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            loss_mmc4 = model(
                vision_x=images,
                lang_x=input_ids.to(device_id),
                attention_mask=attention_mask.to(device_id),
                labels=labels,
            )[0]

            # if loss is nan, skip this batch
            # this hack of skipping the batch is not FSDP-compatible
            if torch.isnan(loss_mmc4):
                print("loss is nan, skipping this batch")
                print("input_ids: ", tokenizer.batch_decode(input_ids))
                print("labels: ", labels)
                print("images: ", images)
                optimizer.zero_grad(set_to_none=True)
                continue

        divided_loss_mmc4 = loss_mmc4 / args.gradient_accumulation_steps
        (divided_loss_mmc4 * args.loss_multiplier_mmc4).backward()

        if (not args.freeze_lm_embeddings) and (
            not args.fsdp or args.fsdp_use_orig_params
        ):
            # Mask gradients for input embeddings s.t. we only update the added tokens <image> and <|endofchunk|>
            if args.fsdp:
                embed_grad = model.lang_encoder.get_input_embeddings().weight.grad
            else:
                embed_grad = (
                    model.module.lang_encoder.get_input_embeddings().weight.grad
                )
            zero_mask = torch.zeros_like(embed_grad)
            zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
            zero_mask[endofchunk_token_id] = torch.ones_like(
                zero_mask[endofchunk_token_id]
            )
            if args.fsdp:
                model.lang_encoder.get_input_embeddings().weight.grad = (
                    embed_grad * zero_mask
                )
            else:
                model.module.lang_encoder.get_input_embeddings().weight.grad = (
                    embed_grad * zero_mask
                )

        # clip gradient norm
        if args.fsdp:
            """
            The way we clip gradients with FSDP is different than the non-FSDP case,
            because during FSDP, gradient norms are computed over certain submodules,
            rather than the entire model.
            At least for OPT-125M, this didn't seem to make a difference in performance.
            """
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            # rank 0 logging
            if args.rank == 0 and args.report_to_wandb:
                laion_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    * args.world_size
                    / step_time_m.val
                )
                laion_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    / step_time_m.val
                )
                c4_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_mmc4
                    * args.world_size
                    / step_time_m.val
                )
                c4_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_mmc4
                    / step_time_m.val
                )
                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "laion_samples_per_second": laion_samples_per_second,
                        "laion_samples_per_second_per_gpu": laion_samples_per_second_per_gpu,
                        "c4_samples_per_second": c4_samples_per_second,
                        "c4_samples_per_second_per_gpu": c4_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_laion": loss_laion.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )
                wandb.log(
                    {"loss_mmc4": loss_mmc4.item(), "global_step": global_step},
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss LAION: {loss_laion.item():.3f} // Loss MMC4: {loss_mmc4.item():.3f}"
            )



## new train one epoch 
# Assume FSDP setup, custom_loader, and other setups are done as per your requirements.
#here device _id would be rank 
def train_one_epoch_new(args, model, epoch, custom_loader, tokenizer, optimizer, lr_scheduler, device_id, wandb):
    model.train()
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for num_steps, batch in enumerate(custom_loader):
        data_time_m.update(time.time() - end)
        
        with torch.cuda.amp.autocast(enabled=args.precision != "fp32"):
            images, input_ids, attention_mask, labels = prepare_batch(batch, tokenizer, device_id)
            loss = model(images, input_ids, attention_mask, labels)[0] / args.gradient_accumulation_steps

        loss.backward()
        
        if (num_steps + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping and optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if dist.get_rank() == 0:
                # Log to WandB or any other logger
                wandb.log({"loss": loss.item(), "epoch": epoch, "step": num_steps})

        step_time_m.update(time.time() - end)
        end = time.time()

    if dist.get_rank() == 0:
        print(f"Epoch {epoch+1} completed. Final Loss: {loss.item():.4f}")

# Helper function to prepare batch data
def prepare_batch(batch, tokenizer, device_id):
    images, input_ids, attention_mask = (
        batch[0].to(device_id, non_blocking=True),
        batch[1].to(device_id, non_blocking=True),
        batch[2].to(device_id, non_blocking=True),
    )
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    labels = labels.to(device_id)
    return images, input_ids, attention_mask, labels
