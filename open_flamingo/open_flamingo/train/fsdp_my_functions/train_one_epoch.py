#this is from chatgpt
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





#this was written before 
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
