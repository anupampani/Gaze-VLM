from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from train_utils_attention import evaluate_model_blip
from data import get_dataset_gaze_attention
import open_clip
from transformers import AutoProcessor, BlipForImageTextRetrieval


def load_blip2_model_and_processor(model_name="Salesforce/blip2-opt-2.7b", device="cuda", dtype=torch.float16):
    """
    Load BLIP-2 model and processor for evaluation.
    """
    print("Loading BLIP-2 model...")
    clip_vision_encoder_path="ViT-L-14"
    clip_vision_encoder_pretrained="openai"
    processor = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
    model = AutoProcessor.from_pretrained(
        "Salesforce/blip-itm-base-coco",
        load_in_8bit=True,  # Enable 8-bit inference for memory efficiency
        device_map={"": 0},  # Map everything to CUDA:0
        torch_dtype=dtype
    )
    
    print("BLIP-2 model and processor loaded successfully.")
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
    clip_vision_encoder_path,
    pretrained=clip_vision_encoder_pretrained
    )
    print("Model and tokenizer loaded successfully.")
    return model, processor,image_processor


if __name__ == "__main__":
    # Paths and configurations
    model_name = "Salesforce/blip2-opt-2.7b"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    generation_config = dict(max_new_tokens=512, do_sample=False)

    # Load BLIP-2 model and processor
    model, processor,image_processor = load_blip2_model_and_processor(model_name, device, dtype)

    # Load test dataset
    test_dataset_info = get_dataset_gaze_attention(processor,
        image_processor,  # Replace `clip_processor` with BLIP-2 processor
        split="test",
        base_path="/home/pani3/dataset_agg"  # Adjust path to your dataset
    )
    test_loader = test_dataset_info.dataloader

    # Evaluate the BLIP-2 model
    print("Starting evaluation with BLIP-2...")
    evaluate_model_blip(test_loader, model, processor, None, generation_config)  # Pass BLIP-2 model and processor
    print("Evaluation completed.")
