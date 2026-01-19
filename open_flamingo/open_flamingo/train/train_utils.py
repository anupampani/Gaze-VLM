import time
import numpy as np
from contextlib import suppress
import torch
from tqdm import tqdm
from torch.distributed import all_reduce, ReduceOp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
import torch.distributed as dist
from torch.distributed.fsdp.api import FullOptimStateDictConfig
import os
import wandb
from einops import rearrange
import re
from bert_score import score
from evaluation_utils import sentence_transformers_similarity
# from sacrebleu.metrics import BLEU
# from rouge import Rouge
from rouge_score import rouge_scorer

import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


import torch
import numpy as np
import nltk
from nltk.translate.meteor_score import meteor_score

nltk.download('wordnet')  # Required for METEOR's synonym matching

def score_meteor_base(model, test_loader, tokenizer, device_id):
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    batch_meteor_scores = []

    model.eval()
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            for num_steps, batch in enumerate(test_loader):
                batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
                images, input_ids, attention_mask, labels = prepare_batch(batch, tokenizer, device_id,media_token_id,endofchunk_token_id)
                outputs = model(images, input_ids, attention_mask, labels)
                logits = outputs.logits 
                token_ids = logits_to_token_ids(logits)
                predicted_text = decode_token_ids_new(tokenizer, token_ids)
                ground_truth = decode_token_ids_new(tokenizer, input_ids)

                # Calculate METEOR scores for each sentence and find the average for the mini-batch
                minibatch_meteor_scores = []

                for pred_text, gt_text in zip(predicted_text, ground_truth):
                    if pred_text.strip():  # Ensure the predicted text is not empty
                        # Tokenize the predicted text and ground truth (split by spaces)
                        hypothesis = pred_text.split()
                        reference = [gt_text.split()]
                        
                        # Compute METEOR score
                        meteor = meteor_score(reference, hypothesis)
                        minibatch_meteor_scores.append(meteor)

                # Ensure non-empty minibatch_meteor_scores before averaging
                if minibatch_meteor_scores:
                    avg_meteor_score = np.mean(minibatch_meteor_scores)
                else:
                    avg_meteor_score = 0.0
                batch_meteor_scores.append(avg_meteor_score)

                # Optional: Print the current step's METEOR score if running on the main device and every 10 steps
                if device_id == 0 and (num_steps % 10 == 0):
                    print(f"Step {num_steps + 1} Average METEOR Score: {avg_meteor_score}")

    # Compute the average METEOR score over all batches
    final_meteor_score = np.mean(batch_meteor_scores)

    # Create a formatted string for the score
    result_string = f"Final METEOR Score: {final_meteor_score:.4f}\n"

    return result_string





def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_mp_policy_dtype(precision: str):
    if "bfloat16" in precision or "bf16" in precision:
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    else:
        return torch.float32


def get_autocast(precision, cache_enabled=True):
    if precision == "amp":
        return torch.cuda.amp.autocast(cache_enabled=cache_enabled)
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(
            dtype=torch.bfloat16, cache_enabled=cache_enabled
        )
    else:
        return suppress

# Helper function to prepare batch data
def prepare_batch(batch, tokenizer, device_id,media_token_id,endofchunk_token_id):
    images, input_ids, attention_mask = (
        batch[0].to(device_id, non_blocking=True),
        batch[1].to(device_id, non_blocking=True),
        batch[2].to(device_id, non_blocking=True),
    )
    labels = input_ids.clone()
    labels[labels == media_token_id] = -100
    labels[labels == endofchunk_token_id] = -100
    labels[labels == tokenizer.pad_token_id] = -100
    labels = labels.to(device_id)
    return images, input_ids, attention_mask, labels

def prepare_batch2(batch, tokenizer, device_id, media_token_id, endofchunk_token_id):
    images, input_ids, attention_mask = (
        batch[0].to(device_id, non_blocking=True),
        batch[1].to(device_id, non_blocking=True),
        batch[2].to(device_id, non_blocking=True),
    )

    # Create a copy of input_ids for labels
    labels = input_ids.clone()

    # Mask <image>, <|endofchunk|>, and padding tokens in the labels for loss calculation
    labels[labels == media_token_id] = -100
    labels[labels == endofchunk_token_id] = -100
    labels[labels == tokenizer.pad_token_id] = -100

    # Convert tokens back to their string representations
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # Dynamically mask numerical tokens (gaze points) in the labels
    for i, token in enumerate(input_tokens):
        if token.isdigit() or re.match(r'^\d+$', token):  # Check if token is a number
            labels[0, i] = -100  # Mask the token in the labels

    labels = labels.to(device_id)
    
    return images, input_ids, attention_mask, labels


def train_one_epoch_v3(args, model, epoch, custom_loader, tokenizer, optimizer, lr_scheduler, device_id, wandb):
    # step_time_m = AverageMeter()
    # data_time_m = AverageMeter()
    total_loss = 0.0  # Track total loss across all batches
    total_samples = 0  # Track total samples processed
    end = time.time()

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()

    for num_steps, batch in enumerate(custom_loader):
        # data_time_m.update(time.time() - end)
        batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor - should be 32

        with torch.cuda.amp.autocast(enabled=args.precision != "fp32"):
            images, input_ids, attention_mask, labels = prepare_batch2(batch, tokenizer, device_id,media_token_id,endofchunk_token_id)
            outputs = model(images, input_ids, attention_mask, labels)
            #change-made: here
            loss = outputs.loss / args.gradient_accumulation_steps
            #loss = outputs / args.gradient_accumulation_steps

        loss.backward()
        torch.cuda.empty_cache()        
        if (num_steps + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            

        # Aggregate loss across all processes
        loss_tensor = torch.tensor([loss.item() * batch_size], device=device_id)
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        total_loss += loss_tensor.item()
        total_samples += batch_size * torch.distributed.get_world_size()
        if device_id==0 and (num_steps % 50 == 0):
            print(
                f"Step {num_steps+1} of epoch {epoch}/{args.epochs} complete. Loss is: {loss.item():.3f}"
            )
    # if(epoch % args.step_size==0):
    lr_scheduler.step()
    avg_loss = total_loss / total_samples
    if torch.distributed.get_rank() == 0:
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        wandb.log({"loss_epoch":avg_loss})
        # logits = outputs.logits 
        # token_ids = logits_to_token_ids(logits)
        # predicted_text = decode_token_ids(tokenizer,token_ids)
        # new_text = decode_token_ids(tokenizer,input_ids)
        # for pred_text, gt_text in zip(predicted_text, new_text):
        #     i=1
        #     print(pred_text)
        #     # print(gt_text)
        #     # wandb.log({"predicted_text":wandb.Table(data=pred_text,columns=["Text"])})
        #     if(i==1):
        #         break
    torch.cuda.empty_cache()
    return avg_loss



def score_rouge_base(model, test_loader, tokenizer, device_id):
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    batch_rouge_l_scores = []

    model.eval()
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            for num_steps, batch in enumerate(test_loader):
                batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
                images, input_ids, attention_mask, labels = prepare_batch(batch, tokenizer, device_id,media_token_id,endofchunk_token_id)
                outputs = model(images, input_ids, attention_mask, labels)
                logits = outputs.logits 
                token_ids = logits_to_token_ids(logits)
                predicted_text = decode_token_ids_new(tokenizer, token_ids)
                ground_truth = decode_token_ids_new(tokenizer, input_ids)

                # Calculate ROUGE scores for each sentence and find the average for the mini-batch
                minibatch_rouge_l_scores = {
                    "precision": [],
                    "recall": [],
                    "fmeasure": []
                }
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

                for pred_text, gt_text in zip(predicted_text, ground_truth):
                    rouge_scores = scorer.score(gt_text, pred_text)
                    minibatch_rouge_l_scores["precision"].append(rouge_scores['rougeL'].precision)
                    minibatch_rouge_l_scores["recall"].append(rouge_scores['rougeL'].recall)
                    minibatch_rouge_l_scores["fmeasure"].append(rouge_scores['rougeL'].fmeasure)

                for key in minibatch_rouge_l_scores:
                    minibatch_rouge_l_scores[key] = np.mean(minibatch_rouge_l_scores[key])
                batch_rouge_l_scores.append(minibatch_rouge_l_scores)

                # Optional: Print the current step's ROUGE-L scores if running on the main device and every 10 steps
                if device_id == 0 and (num_steps % 10 == 0):
                    print(f"Step {num_steps + 1} Average ROUGE-L Precision: {minibatch_rouge_l_scores['precision']}")
                    print(f"Step {num_steps + 1} Average ROUGE-L Recall: {minibatch_rouge_l_scores['recall']}")
                    print(f"Step {num_steps + 1} Average ROUGE-L F1: {minibatch_rouge_l_scores['fmeasure']}")

    # Compute the average ROUGE-L scores over all batches
    final_rouge_l_scores = {
        "precision": np.mean([score["precision"] for score in batch_rouge_l_scores]),
        "recall": np.mean([score["recall"] for score in batch_rouge_l_scores]),
        "fmeasure": np.mean([score["fmeasure"] for score in batch_rouge_l_scores])
    }

    # Create a formatted string for the scores
    result_string = (
        f"Final ROUGE-L Precision: {final_rouge_l_scores['precision']:.4f}\n"
        f"Final ROUGE-L Recall: {final_rouge_l_scores['recall']:.4f}\n"
        f"Final ROUGE-L F1: {final_rouge_l_scores['fmeasure']:.4f}\n"
    )

    return result_string




def score_bleu_base(model, test_loader, tokenizer, device_id):
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    batch_bleu_scores = []

    model.eval()
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            for num_steps, batch in enumerate(test_loader):
                batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor
                images, input_ids, attention_mask, labels = prepare_batch(batch, tokenizer, device_id,media_token_id,endofchunk_token_id)
                outputs = model(images, input_ids, attention_mask, labels)
                logits = outputs.logits 
                token_ids = logits_to_token_ids(logits)
                predicted_text = decode_token_ids_new(tokenizer, token_ids)
                ground_truth = decode_token_ids_new(tokenizer, input_ids)

                # Calculate BLEU scores for each sentence and find the average for the mini-batch
                minibatch_bleu_scores = []

                for pred_text, gt_text in zip(predicted_text, ground_truth):
                    reference = [gt_text.split()]  # Ground truth as list of words
                    hypothesis = pred_text.split()  # Predicted sentence as list of words
                    smoothing_fn = SmoothingFunction().method4
                    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_fn)
                    minibatch_bleu_scores.append(bleu_score)

                # Average the BLEU score for the mini-batch
                avg_bleu_score = np.mean(minibatch_bleu_scores)
                batch_bleu_scores.append(avg_bleu_score)

                # Optional: Print the current step's BLEU score if running on the main device and every 10 steps
                if device_id == 0 and (num_steps % 10 == 0):
                    print(f"Step {num_steps + 1} Average BLEU Score: {avg_bleu_score}")

    # Compute the average BLEU score over all batches
    final_bleu_score = np.mean(batch_bleu_scores)

    # Create a formatted string for the score
    result_string = f"Final BLEU Score: {final_bleu_score:.4f}\n"

    return result_string



# Helper function to prepare batch data
def prepare_batch_gaze(batch, tokenizer, device_id,image_token_id,gaze_token_id,endofchunk_token_id):
    images,input_ids, attention_mask = (
        batch[0].to(device_id, non_blocking=True),
        batch[1].to(device_id, non_blocking=True),
        batch[2].to(device_id, non_blocking=True),
    )
    labels = input_ids.clone()
    labels[labels == image_token_id] = -100
    labels[labels == gaze_token_id] = -100
    labels[labels == endofchunk_token_id] = -100
    labels[labels == tokenizer.pad_token_id] = -100
    labels = labels.to(device_id)
    return images,input_ids, attention_mask, labels

def train_one_epoch_gaze(args, model, epoch, custom_loader, tokenizer, optimizer, lr_scheduler, device_id, wandb):
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    total_loss = 0.0  # Track total loss across all batches
    total_samples = 0  # Track total samples processed
    end = time.time()

    #what does this do, check
    image_token_id = tokenizer("<image>", add_special_tokens=True)["input_ids"][-1]
    gaze_token_id = tokenizer("<gaze>", add_special_tokens=True)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=True)[
        "input_ids"
    ][-1]



    model.train()

    for num_steps, batch in enumerate(custom_loader):
        # data_time_m.update(time.time() - end)
        batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor - should be 32

        with torch.cuda.amp.autocast(enabled=args.precision != "fp32"):
            images, input_ids, attention_mask, labels = prepare_batch_gaze(batch, tokenizer, device_id,image_token_id,gaze_token_id,endofchunk_token_id)
            outputs = model(images, input_ids, attention_mask, labels)
            loss = outputs.loss / args.gradient_accumulation_steps
            #loss = outputs / args.gradient_accumulation_steps
        loss.backward()
        torch.cuda.empty_cache()  
        
        if (num_steps + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            

        # Aggregate loss across all processes
        loss_tensor = torch.tensor([loss.item() * batch_size], device=device_id)
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        total_loss += loss_tensor.item()
        total_samples += batch_size * torch.distributed.get_world_size()
        if device_id==0 and (num_steps % 50 == 0):
            print(
                f"Step {num_steps+1} of epoch {epoch}/{args.epochs} complete. Loss is: {loss.item():.3f}"
            )
            # Assuming your model's output is logits and you're dealing with a classification task




        # Calculate average loss for the current step across all processes
        # step_avg_loss = loss_tensor.item() / (batch_size * torch.distributed.get_world_size())
        # if torch.distributed.get_rank() == 0:
        #     print(f"Step {num_steps+1}, Epoch {epoch}, Average Step Loss: {step_avg_loss:.4f}")

        # step_time_m.update(time.time() - end)
        # end = time.time()

    # Calculate average loss across all processes and samples
    # if(epoch % args.step_size==0):
    lr_scheduler.step()
    avg_loss = total_loss / total_samples
    if torch.distributed.get_rank() == 0:
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        wandb.log({"loss_epoch":avg_loss})
        logits = outputs.logits 
        token_ids = logits_to_token_ids(logits)
        predicted_text = decode_token_ids(tokenizer,token_ids)
        new_text = decode_token_ids(tokenizer,input_ids)
        for pred_text, gt_text in zip(predicted_text, new_text):
            i=1
            print(pred_text)
            # wandb.log({"predicted_text":wandb.Table(data=pred_text,columns=["Text"])})
            if(i==1):
                break

        
    torch.cuda.empty_cache()
    return avg_loss

#new validation function which includes the synchronisation from all processes
def validate(args, model, valid_loader, tokenizer, device_id):
    # model.eval()
    total_loss = 0.0  # Aggregate loss for calculating average later
    total_samples = 0  # Count of total samples processed

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    gaze_token_id = tokenizer("<gaze>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.eval()
    with torch.no_grad():
        for num_steps,batch in enumerate(valid_loader):
            batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
            images, input_ids, attention_mask, labels = prepare_batch(batch, tokenizer, device_id,media_token_id,endofchunk_token_id)
            outputs = model(images, input_ids, attention_mask, labels)
            loss = outputs.loss / args.gradient_accumulation_steps

            # Multiply loss by batch size to get total loss for this batch (before reduction)
            loss_tensor = torch.tensor([loss.item() * batch_size], device=device_id)
            
            # Reduce the loss across all processes
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            
            # Add to total loss and samples
            total_loss += loss_tensor.item()
            total_samples += batch_size * torch.distributed.get_world_size()  # Adjust for actual samples across all processes
            if device_id==0 and (num_steps % 50 == 0):
                print(
                    f"Step {num_steps+1}  Loss is: {loss.item():.3f}"
                )
    # Calculate average loss across all samples and processes
    avg_loss = total_loss / total_samples

    # Only rank 0 prints, but all ranks return the average loss for possible further processing
    if torch.distributed.get_rank() == 0:
        print(f"Validation Loss: {avg_loss:.4f}")
    
    return avg_loss


def validate_gaze(args, model, valid_loader, tokenizer, device_id):
    # model.eval()
    total_loss = 0.0  # Aggregate loss for calculating average later
    total_samples = 0  # Count of total samples processed
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    gaze_token_id = tokenizer("<gaze>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]
    model.eval()
    with torch.no_grad():
        for num_steps,batch in enumerate(valid_loader):
            batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
            images, input_ids, attention_mask, labels = prepare_batch_gaze(batch, tokenizer, device_id,media_token_id,gaze_token_id,endofchunk_token_id)
            outputs = model(images, input_ids, attention_mask, labels)
            loss = outputs.loss / args.gradient_accumulation_steps

            # Multiply loss by batch size to get total loss for this batch (before reduction)
            loss_tensor = torch.tensor([loss.item() * batch_size], device=device_id)
            
            # Reduce the loss across all processes
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            
            # Add to total loss and samples
            total_loss += loss_tensor.item()
            total_samples += batch_size * torch.distributed.get_world_size()  # Adjust for actual samples across all processes

            if device_id==0 and (num_steps % 50 == 0):
                print(
                    f"Step {num_steps+1}  Loss is: {loss.item():.3f}"
                )
    # Calculate average loss across all samples and processes
    avg_loss = total_loss / total_samples

    # Only rank 0 prints, but all ranks return the average loss for possible further processing
    if torch.distributed.get_rank() == 0:
        print(f"Validation Loss: {avg_loss:.4f}")
    
    return avg_loss


def logits_to_token_ids(logits):
    # Select the token with the highest logit value at each time step
    predicted_token_ids = torch.argmax(logits, dim=-1)
    return predicted_token_ids

# def decode_token_ids(tokenizer, token_ids):
#     # Convert the tensor of token IDs to a list if it's not already in list form
#     # if isinstance(token_ids, torch.Tensor):
#     #     token_ids = token_ids.tolist()

#     # Decode the list of token IDs to a string
#     # for id in token_ids:
#     #     text=''.join(tokenizer.decode(id,skip_special_tokens=True))
#     decoded_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]

#     return decoded_texts

# def decode_token_ids(tokenizer, token_ids):
#     # First, decode each set of token IDs in the batch into text
#     decoded_texts = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in token_ids]
#     # Then, enclose each decoded text in quotation marks
#     quoted_texts = ['"' + text + '"' for text in decoded_texts]
#     return quoted_texts

# def decode_token_ids(tokenizer, token_ids):
#     # First, decode each set of token IDs in the batch into text
#     decoded_texts = []
#     for ids in token_ids:
#         if -1 in ids or -100 in ids:
#             continue
#         decoded_text = tokenizer.decode(ids, skip_special_tokens=True).strip()
#         decoded_texts.append(decoded_text)

#     # Then, enclose each decoded text in quotation marks
#     quoted_texts = ['"' + text + '"' for text in decoded_texts]
#     return quoted_texts
def decode_token_ids(tokenizer, token_ids):
    # First, decode each set of token IDs in the batch into text
    decoded_texts = []
    for ids in token_ids:
        decoded_tokens = []
        for token_id in ids:
            if token_id in [-1, -100]:
                continue
            decoded_token = tokenizer.decode([token_id], skip_special_tokens=True).strip()
            decoded_tokens.append(decoded_token)
        
        # Remove repetitive tokens at the end of the sentence
        if len(decoded_tokens) > 1:
            last_token = decoded_tokens[-1]
            if all(token == last_token for token in decoded_tokens[:-1]):
                decoded_tokens = decoded_tokens[:-1]
        
        decoded_text = " ".join(decoded_tokens)
        decoded_texts.append(decoded_text)

    # Then, enclose each decoded text in quotation marks
    quoted_texts = ['"' + text + '"' for text in decoded_texts]
    return quoted_texts

# def decode_token_ids_new(tokenizer, token_ids):

#     # First, decode each set of token IDs in the batch into text
#     decoded_texts = []
#     for ids in token_ids:
#         decoded_tokens = []
#         for token_id in ids:
#             if token_id in [-1, -100]:
#                 continue
#             decoded_token = tokenizer.decode([token_id], skip_special_tokens=True).strip()
#             decoded_tokens.append(decoded_token)
        
#         decoded_text = " ".join(decoded_tokens)
        
#         # Remove the specified substring from the decoded text
#         decoded_text = decoded_text.replace(":", "")
#         decoded_text = decoded_text.replace("!", "")
#         decoded_text = decoded_text.replace("-", "")
#         # decoded_text = decoded_text.replace(remove2, "")
        
#         decoded_texts.append(decoded_text)

#     # Then, enclose each decoded text in quotation marks
#     quoted_texts = ['"' + text + '"' for text in decoded_texts]
#     return quoted_texts


def decode_token_ids_new(tokenizer, token_ids):

    # First, decode each set of token IDs in the batch into text
    decoded_texts = []
    for ids in token_ids:
        decoded_tokens = []
        prev_token = None
        for token_id in ids:
            if token_id in [-1, -100]:
                continue
            decoded_token = tokenizer.decode([token_id], skip_special_tokens=True).strip()
            if decoded_token != prev_token:
                decoded_tokens.append(decoded_token)
                prev_token = decoded_token
        
        decoded_text = " ".join(decoded_tokens)
        
        # Remove the specified substring from the decoded text
        decoded_text = decoded_text.replace(":", "")
        decoded_text = decoded_text.replace("!", "")
        decoded_text = decoded_text.replace("-", "")
        decoded_text = decoded_text.replace("", "")
        # decoded_text = decoded_text.replace(remove2, "")
        
        decoded_texts.append(decoded_text)

    # Then, enclose each decoded text in quotation marks
    quoted_texts = ['"' + text + '"' for text in decoded_texts]
    return quoted_texts

def baseline_score(model,test_loader,tokenizer,device_id):
    
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    gaze_token_id = tokenizer("<gaze>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    batch_scores=[]

    model.eval()
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            for num_steps, batch in enumerate(test_loader):
                batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
                images, input_ids, attention_mask, labels = prepare_batch(batch, tokenizer, device_id,media_token_id,endofchunk_token_id)
                outputs = model(images, input_ids, attention_mask,labels)
                logits = outputs.logits 
                token_ids = logits_to_token_ids(logits)
                predicted_text = decode_token_ids(tokenizer,token_ids)
                ground_truth = decode_token_ids(tokenizer,input_ids)

                if torch.distributed.get_rank() ==0:
                    score= np.mean(sentence_transformers_similarity(predicted_text,ground_truth))
                    batch_scores.append(score)
                if device_id==0 and (num_steps % 10 == 0):
                    print(
                        f"Step {num_steps+1} Loss is: {score}"
                    )

    return batch_scores



import numpy as np
import torch

def baseline_score2(model, test_loader, tokenizer, device_id):
    
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    gaze_token_id = tokenizer("<gaze>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    batch_scores = []

    model.eval()
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            for num_steps, batch in enumerate(test_loader):
                batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
                images, input_ids, attention_mask, labels = prepare_batch(batch, tokenizer, device_id,media_token_id,endofchunk_token_id)
                outputs = model(images, input_ids, attention_mask,labels)
                logits = outputs.logits 
                token_ids = logits_to_token_ids(logits)
                predicted_text = decode_token_ids_new(tokenizer,token_ids)
                ground_truth = decode_token_ids_new(tokenizer,input_ids)

            # Calculate similarities for each sentence and find the average for the mini-batch
                minibatch_scores = []
                for pred_text, gt_text in zip(predicted_text, ground_truth):
                    score = sentence_transformers_similarity([pred_text], [gt_text])
                    minibatch_scores.append(score)

                minibatch_average_score = np.mean(minibatch_scores)
                batch_scores.append(minibatch_average_score)

            # Optional: Print the current step's loss if running on the main device and every 10 steps
                if device_id == 0 and (num_steps % 10 == 0):
                    print(f"Step {num_steps + 1} Average Score: {minibatch_average_score}")

    # Compute the average score over all batches
    final_average_score = np.mean(batch_scores)
    if device_id==0:
        print(f"Average Score: {final_average_score}")
        
    return final_average_score



# def baseline_rogue(model, test_loader, tokenizer, device_id):
#     rouge_scorer = Rouge()
#     bleu_scorer = BLEU()

#     media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
#     gaze_token_id = tokenizer("<gaze>", add_special_tokens=False)["input_ids"][-1]
#     endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
#         "input_ids"
#     ][-1]

#     batch_scores1 = []
#     batch_scores2 = []
#     model.eval()
#     with torch.inference_mode():
#         with torch.cuda.amp.autocast():
#             for num_steps, batch in enumerate(test_loader):
#                 batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
#                 images, input_ids, attention_mask, labels = prepare_batch(batch, tokenizer, device_id,media_token_id,endofchunk_token_id)
#                 outputs = model(images, input_ids, attention_mask,labels)
#                 logits = outputs.logits 
#                 token_ids = logits_to_token_ids(logits)
#                 predicted_text = decode_token_ids_new(tokenizer,token_ids)
#                 ground_truth = decode_token_ids_new(tokenizer,input_ids)

#             # Calculate similarities for each sentence and find the average for the mini-batch
#                 minibatch_scores1 = []
#                 minibatch_scores2 = []
                
#                 for pred_text, gt_text in zip(predicted_text, ground_truth):
#                     score1 = bleu_scorer.sentence_score(hypothesis=pred_text, references=[gt_text])
#                     minibatch_scores1.append(score1)
#                     score2= rouge_scorer.get_scores(hyps=pred_text,refs=gt_text)
#                     minibatch_scores2.append(score2[0]["rouge-l"]["f"])

#                 minibatch_average_score = np.mean(minibatch_scores1)
#                 batch1_scores.append(minibatch_average_score)
#                 minibatch_average_score = np.mean(minibatch_scores2)
#                 batch2_scores.append(minibatch_average_score)

#             # Optional: Print the current step's loss if running on the main device and every 10 steps
#                 if device_id == 0 and (num_steps % 10 == 0):
#                     print(f"Step {num_steps + 1} Average Score: {minibatch_average_score}")

#     # Compute the average score over all batches
#     final_average_score1 = np.mean(batch_scores1)
#     final_average_score2 = np.mean(batch_scores2)
#     if device_id==0:
#         print(f"Average Score: {final_average_score1}")
        
#     return final_average_score1,final_average_score2

# def gaze_rogue(args, model, test_loader, tokenizer, device_id):

#     rouge_scorer = Rouge()
#     bleu_scorer = BLEU()
#     model.eval()
#     total_loss = 0.0  # Aggregate loss for calculating average later
#     total_samples = 0  # Count of total samples processed
#     media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
#     gaze_token_id = tokenizer("<gaze>", add_special_tokens=False)["input_ids"][-1]
#     endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
#         "input_ids"
#     ][-1]


#     batch_scores1 = []
#     batch_scores2 = []
#     model.eval()
#     with torch.inference_mode():
#         with torch.cuda.amp.autocast():
#             for num_steps, batch in enumerate(test_loader):
#                 batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
#                 images, input_ids, attention_mask, labels = prepare_batch_gaze(batch, tokenizer, device_id,image_token_id,gaze_token_id,endofchunk_token_id)
#                 outputs = model(images, input_ids, attention_mask,labels)
#                 logits = outputs.logits 
#                 token_ids = logits_to_token_ids(logits)
#                 predicted_text = decode_token_ids_new(tokenizer,token_ids)
#                 ground_truth = decode_token_ids_new(tokenizer,input_ids)

#             # Calculate similarities for each sentence and find the average for the mini-batch
#                 minibatch_scores1 = []
#                 minibatch_scores2 = []
                
#                 for pred_text, gt_text in zip(predicted_text, ground_truth):
#                     score1 = blue_scorer.sentence_score(hypothesis=pred_text, references=[gt_text])
#                     minibatch_scores1.append(score1)
#                     score2= rouge_scorer.get_scores(hyps=pred_text,refs=gt_text)
#                     minibatch_scores2.append(score2[0]["rouge-l"]["f"])

#                 minibatch_average_score = np.mean(minibatch_scores1)
#                 batch1_scores.append(minibatch_average_score)
#                 minibatch_average_score = np.mean(minibatch_scores2)
#                 batch2_scores.append(minibatch_average_score)

#             # Optional: Print the current step's loss if running on the main device and every 10 steps
#                 if device_id == 0 and (num_steps % 10 == 0):
#                     print(f"Step {num_steps + 1} Average Score: {minibatch_average_score}")

#     # Compute the average score over all batches
#     final_average_score1 = np.mean(batch_scores1)
#     final_average_score2 = np.mean(batch_scores2)
#     if device_id==0:
#         print(f"Average Score: {final_average_score1}")
        
    # return final_average_score1,final_average_score2





def gaze_score(model,test_loader,tokenizer,device_id):
    
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    gaze_token_id = tokenizer("<gaze>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    batch_scores=[]

    model.eval()
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            for num_steps, batch in enumerate(test_loader):
                batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
                images, input_ids, attention_mask, labels = prepare_batch_gaze(batch, tokenizer, device_id,media_token_id,gaze_token_id,endofchunk_token_id)
                outputs = model(images, input_ids, attention_mask,labels)
                logits = outputs.logits 
                token_ids = logits_to_token_ids(logits)
                predicted_text = decode_token_ids(tokenizer,token_ids)
                ground_truth = decode_token_ids(tokenizer,input_ids)

                if torch.distributed.get_rank() ==0:
                    score= np.mean(sentence_transformers_similarity(predicted_text,ground_truth))
                    batch_scores.append(score)
                if device_id==0 and (num_steps % 10 == 0):
                    print(
                        f"Step {num_steps+1} Loss is: {score}"
                    )

    return batch_scores

def gaze_dist(model,test_loader,tokenizer,device_id):
    
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    gaze_token_id = tokenizer("<gaze>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    batch_scores=[]

    model.eval()
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            for num_steps, batch in enumerate(test_loader):
                batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
                images, input_ids, attention_mask, labels = prepare_batch_gaze(batch, tokenizer, device_id,media_token_id,gaze_token_id,endofchunk_token_id)
                outputs = model(images, input_ids, attention_mask,labels)
                logits = outputs.logits 
                token_ids = logits_to_token_ids(logits)
                predicted_text = decode_token_ids(tokenizer,token_ids)
                ground_truth = decode_token_ids(tokenizer,input_ids)

                if torch.distributed.get_rank() ==0:
                    score= np.mean(sentence_transformers_similarity(predicted_text,ground_truth))
                    batch_scores.append(score)
                if device_id==0 and (num_steps % 10 == 0):
                    print(
                        f"Step {num_steps+1} Loss is: {score}"
                    )

    return batch_scores

def evaluate_base(args, model, test_loader, tokenizer, device_id):
    model.eval()
    total_loss = 0.0  # Aggregate loss for calculating average later
    total_samples = 0  # Count of total samples processed
    correct_predictions = 0

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    gaze_token_id = tokenizer("<gaze>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    with torch.no_grad():
        for num_steps,batch in enumarate(test_loader):
            batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
        with torch.cuda.amp.autocast(enabled=args.precision != "fp32"):
            images, input_ids, attention_mask, labels = prepare_batch(batch, tokenizer, device_id,media_token_id,endofchunk_token_id)
            outputs = model(images, input_ids, attention_mask, labels)
            loss = outputs.loss / args.gradient_accumulation_steps
        torch.cuda.empty_cache() 
        loss_tensor = torch.tensor([loss.item() * batch_size], device=device_id)
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        total_loss += loss_tensor.item()
        total_samples += batch_size * torch.distributed.get_world_size()
        if device_id==0 and (num_steps % 50 == 0):
            print(
                f"Step {num_steps+1} complete. Loss is: {loss.item():.3f}"
            )
    avg_loss = total_loss / total_samples
    if torch.distributed.get_rank() == 0:
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    return avg_loss


def evaluate_gaze(args, model, test_loader, tokenizer, device_id):
    model.eval()
    total_loss = 0.0  # Aggregate loss for calculating average later
    total_samples = 0  # Count of total samples processed
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    gaze_token_id = tokenizer("<gaze>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]
    correct_predictions = 0

    with torch.no_grad():
        for num_steps,batch in enumarate(test_loader):
            batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
        with torch.cuda.amp.autocast(enabled=args.precision != "fp32"):
            images, input_ids, attention_mask, labels = prepare_batch_gaze(batch, tokenizer, device_id,media_token_id,endofchunk_token_id)
            outputs = model(images, input_ids, attention_mask, labels)
            loss = outputs.loss / args.gradient_accumulation_steps
        torch.cuda.empty_cache() 
        loss_tensor = torch.tensor([loss.item() * batch_size], device=device_id)
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        total_loss += loss_tensor.item()
        total_samples += batch_size * torch.distributed.get_world_size()
        if device_id==0 and (num_steps % 50 == 0):
            print(
                f"Step {num_steps+1} complete. Loss is: {loss.item():.3f}"
            )
    avg_loss = total_loss / total_samples
    if torch.distributed.get_rank() == 0:
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    return avg_loss



def gaze_score2(model, test_loader, tokenizer, device_id):
    
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    gaze_token_id = tokenizer("<gaze>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    batch_scores = []

    model.eval()
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            for num_steps, batch in enumerate(test_loader):
                batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
                images, input_ids, attention_mask, labels = prepare_batch_gaze(batch, tokenizer, device_id,media_token_id,gaze_token_id,endofchunk_token_id)
                outputs = model(images, input_ids, attention_mask,labels)
                logits = outputs.logits 
                token_ids = logits_to_token_ids(logits)
                predicted_text = decode_token_ids_new(tokenizer,token_ids)
                ground_truth = decode_token_ids_new(tokenizer,input_ids)

            # Calculate similarities for each sentence and find the average for the mini-batch
                minibatch_scores = []
                for pred_text, gt_text in zip(predicted_text, ground_truth):
                    score = sentence_transformers_similarity([pred_text], [gt_text])
                    minibatch_scores.append(score)

                minibatch_average_score = np.mean(minibatch_scores)
                batch_scores.append(minibatch_average_score)

            # Optional: Print the current step's loss if running on the main device and every 10 steps
                if device_id == 0 and (num_steps % 10 == 0):
                    print(f"Step {num_steps + 1} Average Score: {minibatch_average_score}")

    # Compute the average score over all batches
    final_average_score = np.mean(batch_scores)
    return final_average_score


# def text_generate_debug(model,test_loader,tokenizer,device_id):
    
#     media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
#     gaze_token_id = tokenizer("<gaze>", add_special_tokens=False)["input_ids"][-1]
#     endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
#         "input_ids"
#     ][-1]

#     print("checkpoint 2")
#     model.eval()
#     with torch.inference_mode():
#         with torch.cuda.amp.autocast():
#             for num_steps, batch in enumerate(test_loader):
#                 batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
#                 images, input_ids, attention_mask, labels = prepare_batch_gaze(batch, tokenizer, device_id,media_token_id,gaze_token_id,endofchunk_token_id)
#                 outputs = model(images, input_ids, attention_mask,labels)
#                 logits = outputs.logits 
#                 token_ids = logits_to_token_ids(logits)

#                 predicted_text = decode_token_ids_new(tokenizer,token_ids)
#                 new_text = decode_token_ids_new(tokenizer,input_ids)

#                 # outputs = outputs[:, len(input_ids[0]) :]
#                 # predicted_text = tokenizer.batch_decode(outputs,skip_special_tokens=True)
#                 if torch.distributed.get_rank() ==0:
#                     with open("insert path/open_flamingo/open_flamingo/models/gaze_model/output_10_epochs_with_normal_dataset.txt", "a") as file:
#                         for pred_text, gt_text in zip(predicted_text, new_text):
#                             # Write predicted text and ground truth text to the file
#                                 file.write("Predicted Text: " + pred_text + "\n")
#                                 file.write("Ground Truth Text: " + gt_text + "\n")
#                                 file.write("\n")

#                 if num_steps > 10:
#                     break
#     return predicted_text


def text_generate_debug(model,test_loader,tokenizer,device_id):
    
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    gaze_token_id = tokenizer("<gaze>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    print("checkpoint 2")
    model.eval()
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            for num_steps, batch in enumerate(test_loader):
                # batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
                if(num_steps>655):
                    batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
                    images, input_ids, attention_mask, labels = prepare_batch_gaze(batch, tokenizer, device_id,media_token_id,gaze_token_id,endofchunk_token_id)
                    outputs = model(images, input_ids, attention_mask,labels)
                    logits = outputs.logits 
                    token_ids = logits_to_token_ids(logits)
                    predicted_text = decode_token_ids_new(tokenizer,token_ids)
                    new_text = decode_token_ids_new(tokenizer,input_ids)

                    # outputs = outputs[:, len(input_ids[0]) :]
                    # predicted_text = tokenizer.batch_decode(outputs,skip_special_tokens=True)
                    if torch.distributed.get_rank() ==0:
                        with open("insert path/open_flamingo/open_flamingo/models/gaze_model/output10_epochs_normal_dataset.txt", "a") as file:
                            for pred_text, gt_text in zip(predicted_text, new_text):
                                # Write predicted text and ground truth text to the file
                                    file.write("Predicted Text: " + pred_text + "\n")
                                    file.write("Ground Truth Text: " + gt_text + "\n")
                                    file.write("\n")

                # if num_steps > 10:
                #     break
    return predicted_text



def base_generate_debug(model,test_loader,tokenizer,device_id):
    
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    gaze_token_id = tokenizer("<gaze>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    print("checkpoint 2")
    model.eval()
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            for num_steps, batch in enumerate(test_loader):
                batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
                images, input_ids, attention_mask, labels = prepare_batch(batch, tokenizer, device_id,media_token_id,endofchunk_token_id)
                outputs = model(images, input_ids, attention_mask,labels)
                logits = outputs.logits 
                token_ids = logits_to_token_ids(logits)
                predicted_text = decode_token_ids_new(tokenizer,token_ids)
                new_text = decode_token_ids_new(tokenizer,input_ids)


                # outputs = outputs[:, len(input_ids[0]) :]
                # predicted_text = tokenizer.batch_decode(outputs,skip_special_tokens=True)
                if torch.distributed.get_rank() ==0:
                    # print(predicted_text)
                    # print("BREAK ---------------------------------------------------------------- \n")
                    # print(new_text)
                #     for pred_text, gt_text in zip(predicted_text, new_text):
                #         i=1
                #         print(pre@d_text)
                #         print(gt_text)
                #if num_steps>0:
                #     break
                    with open("insert path/open_flamingo/open_flamingo/models/base_model/output10_epochs_2.txt", "a") as file:
                        for pred_text, gt_text in zip(predicted_text, new_text):
                            # Write predicted text and ground truth text to the file
                                file.write("Predicted Text: " + pred_text + "\n")
                                file.write("Ground Truth Text: " + gt_text + "\n")
                                file.write("\n")

    return predicted_text


def calculate_bert_scores(references, predictions, lang='en', model_type=None):
    """
    Calculate BERTScore.

    :param references: List of reference sentences.
    :param predictions: List of predicted sentences generated by the model.
    :param lang: Language of the text. Default is English ('en'). Use 'zh' for Chinese, etc.
    :param model_type: BERT model to use. Default is None, which uses the default BERT model for the language.
                       Examples include 'bert-base-uncased' for English or 'bert-base-multilingual-cased' for multi-language support.
    :return: Precision, Recall, and F1 scores.
    """
    if model_type is None:
        model_type = 'bert-base-uncased' if lang == 'en' else 'bert-base-multilingual-cased'

    # Calculate BERTScore
    P, R, F1 = score(predictions, references, lang=lang, model_type=model_type, verbose=True)

    return P.mean().item(), R.mean().item(), F1.mean().item()

# Example usage:
# references = ["the quick brown fox jumps over the lazy dog", ...]
# predictions = ["the fast brown fox jumps over the lazy dog", ...]
# precision, recall, f1_score = calculate_bert_scores(references, predictions)
# print("Precision:", precision, "Recall:", recall, "F1 Score:", f1_score)









#evaluation function - to get test results 
def evaluate(args, model, test_loader, tokenizer, device_id):
    model.eval()
    total_loss = 0.0  # Aggregate loss for calculating average later
    total_samples = 0  # Count of total samples processed
    correct_predictions = 0

    with torch.no_grad():
        for batch in test_loader:
            batch_size = batch[0].size(0)  # Assuming batch[0] is your input tensor, should be 32
            images, input_ids, attention_mask, labels = prepare_batch(batch, tokenizer, device_id)
            outputs = model(images, input_ids, attention_mask, labels)
            loss = outputs.loss / args.gradient_accumulation_steps

            # Multiply loss by batch size to get total loss for this batch (before reduction)
            loss_tensor = torch.tensor([loss.item() * batch_size], device=device_id)
            
            # Reduce the loss across all processes
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            
            # Add to total loss and samples
            total_loss += loss_tensor.item()
            total_samples += batch_size * torch.distributed.get_world_size()  # Adjust for actual samples across all processes

    # Calculate average loss across all samples and processes
    avg_loss = total_loss / total_samples

    # Only rank 0 prints, but all ranks return the average loss for possible further processing
    if torch.distributed.get_rank() == 0:
        print(f"test Loss: {avg_loss:.4f}")

    return avg_loss


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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def filter_state_dict_to_trainable(model, state_dict):
    """
    Filters out non-trainable parameters and specific named parameters and modules
    from the state dict. This minimizes checkpoint size and focuses on relevant parameters.
    """
    filtered_state_dict = {}
    for name, param in model.named_parameters():
        if "fsdp" in name or not param.requires_grad:
            continue
        if "embed" in name or isinstance(param, torch.nn.Embedding):
            filtered_state_dict[name] = state_dict[name]
        elif param.requires_grad:
            filtered_state_dict[name] = state_dict[name]

    #Specific parameters and modules to be excluded, based on your requirements
    exclusions = ["lang_encoder.old_decoder_blocks", "lang_encoder.gated_cross_attn_layers", "vision_encoder"]
    for key in list(state_dict.keys()):
        if any(exclusion in key for exclusion in exclusions):
            del state_dict[key]
        else:
            # For parameters that are not excluded, add to the filtered state dict
            if key not in filtered_state_dict:
                filtered_state_dict[key] = state_dict[key]

    return filtered_state_dict

def save_checkpoint(model, optimizer, scheduler, epoch, filepath):
    """
    Saves the model, optimizer, and scheduler states to a checkpoint, with specific
    filtering for FSDP compatibility and minimization of checkpoint size.
    """
    # Synchronize all processes before starting the saving process.
    torch.distributed.barrier()
    model_state = model.state_dict()
    optim_state = FSDP.optim_state_dict(model, optimizer)
    # Ensure only rank 0 saves the checkpoint to avoid redundancy and potential issues.
    if torch.distributed.get_rank() == 0:
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_state = filter_state_dict_to_trainable(model, model_state)
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "scheduler_state_dict": scheduler.state_dict(),
        }

        # Attempt to save the checkpoint and handle potential exceptions.
        try:
            torch.save(checkpoint_dict, filepath)
            print(f"Checkpoint successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving checkpoint to {filepath}: {e}")

    # Barrier after saving to ensure synchronization.
    torch.distributed.barrier()



#this will be used for gaze
def save_checkpoint_gaze(model, optimizer, scheduler, epoch, filepath):
    """
    Saves the model, optimizer, and scheduler states to a checkpoint, with specific
    filtering for FSDP compatibility and minimization of checkpoint size.
    """
    # Synchronize all processes before starting the saving process.
    torch.distributed.barrier()
    FSDP.set_state_dict_type(model,StateDictType.FULL_STATE_DICT,state_dict_config=FullStateDictConfig(rank0_only=True),optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True))

    model_state = model.state_dict()
    optim_state = FSDP.optim_state_dict(model, optimizer)

    # Ensure only rank 0 saves the checkpoint to avoid redundancy and potential issues.


    if torch.distributed.get_rank() == 0:
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_state = filter_state_dict_to_trainable(model, model_state)
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "scheduler_state_dict": scheduler.state_dict(),
        }

        # Attempt to save the checkpoint and handle potential exceptions.
        try:
            torch.save(checkpoint_dict, filepath)
            print(f"Checkpoint successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving checkpoint to {filepath}: {e}")

    # Barrier after saving to ensure synchronization.
    torch.distributed.barrier()