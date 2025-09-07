# -*- coding: utf-8 -*-

# generate_captions.py (Revised to fix processor.patch_size)

import torch
# Use AutoProcessor and LlavaForConditionalGeneration
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import argparse
import glob
import logging
import json
import traceback # For detailed error printing

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
parser = argparse.ArgumentParser(description="Generate medical captions using a PEFT fine-tuned Llava-Llama model.")

# --- Model Paths ---
parser.add_argument(
    "--base_model_path",
    type=str,
    default="xtuner/llava-llama-3-8b-v1_1-transformers", # This seems to be LLaVA v1.5
    help="Path or HuggingFace repo ID of the BASE Llava model.",
)
parser.add_argument(
    "--adapter_repo",
    type=str,
    default="JoVal26/ja-med-clef-model",
    help="HuggingFace repo ID containing the PEFT LoRA adapters for captioning.",
)

# --- Other Arguments ---
parser.add_argument(
    "--image_dir",
    type=str,
    default="./data/test/images",
    help="Directory containing the validation images for caption generation.",
)
parser.add_argument(
    "--output_csv",
    type=str,
    default="submission.csv",
    help="Path to save the output caption submission CSV file.",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Primary device preference ('cuda' or 'cpu'). device_map='auto' takes precedence.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4, # Keep small for debugging
    help="Batch size for inference.",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=100,
    help="Maximum number of new tokens to generate for captions.",
)
parser.add_argument(
    "--merge_adapters",
    action='store_true',
    help="Merge LoRA adapters into the base model before inference.",
)
parser.add_argument(
    "--load_in_4bit",
    action='store_true',
    help="Load the base model in 4-bit.",
)
parser.add_argument(
    "--lora_subfolder",
    type=str,
    default=None,
    help="Subfolder within the adapter_repo where LoRA adapter files are located (if any)."
)
parser.add_argument(
    "--generation_kwargs",
    type=str,
    default='{"do_sample": false}',
    help='JSON string of additional kwargs for the model.generate() method. Example: \'{"do_sample": true, "temperature": 0.7}\'',
)

args = parser.parse_args()

# --- Device Setup ---
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
logging.info(f"Preferred device: {device}")
if args.device == 'cuda' and torch.cuda.is_available():
    logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    logging.info(f"CUDA Available: {torch.cuda.is_available()}")
else:
     logging.info(f"Using device: {device}")

# --- Parse Generation Kwargs ---
try:
    generation_params = json.loads(args.generation_kwargs)
    logging.info(f"Using generation parameters: {generation_params}")
except json.JSONDecodeError as e:
    logging.error(f"Invalid JSON for --generation_kwargs: {args.generation_kwargs}. Error: {e}")
    logging.warning("Falling back to default generation parameters: {'do_sample': False}")
    generation_params = {"do_sample": False}


# --- Quantization Config ---
quantization_config = None
model_dtype = torch.float16 if args.device == 'cuda' and torch.cuda.is_available() and not args.load_in_4bit else None

if args.load_in_4bit:
    try:
        from transformers import BitsAndBytesConfig
        if not torch.cuda.is_available(): raise ValueError("4-bit loading needs CUDA.")
        logging.info("Setting up 4-bit quantization.")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model_dtype = None
    except ImportError:
        logging.warning("BitsAndBytesConfig not found. Install bitsandbytes. Proceeding without 4-bit.")
        args.load_in_4bit = False
        model_dtype = torch.float16 if args.device == 'cuda' and torch.cuda.is_available() else torch.float32
    except Exception as e:
        logging.error(f"Error in 4-bit setup: {e}. Proceeding without.")
        args.load_in_4bit = False
        model_dtype = torch.float16 if args.device == 'cuda' and torch.cuda.is_available() else torch.float32
elif device.type == 'cpu':
    model_dtype = torch.float32
    logging.info("Using CPU, setting model dtype to float32.")

# --- Load Processor using AutoProcessor ---
logging.info(f"Attempting to load processor using AutoProcessor from: {args.base_model_path}")
try:
    # Use AutoProcessor to load the correct processor type for the model
    processor = AutoProcessor.from_pretrained(args.base_model_path)
    logging.info(f"Successfully loaded processor using AutoProcessor. Type: {type(processor)}")

    # --- *** START: Ensure processor.patch_size is set *** ---
    EXPECTED_PATCH_SIZE = 14 # Default for ViT-L/14 used in LLaVA v1.5

    # Check if patch_size exists directly on the processor object and is not None
    if not hasattr(processor, 'patch_size') or processor.patch_size is None:
        logging.warning("Processor object attribute 'patch_size' is missing or None. Attempting to set it.")
        patch_size_found = None
        # Try to get it from the image processor's config (most reliable source)
        if hasattr(processor, 'image_processor') and processor.image_processor is not None:
            if hasattr(processor.image_processor, 'config') and processor.image_processor.config is not None:
                 patch_size_found = getattr(processor.image_processor.config, 'patch_size', None)
                 if patch_size_found:
                     logging.info(f"Found patch_size={patch_size_found} in image_processor.config.")

        # If found and is an integer, assign it to the main processor object
        if patch_size_found and isinstance(patch_size_found, int):
            logging.info(f"Setting processor.patch_size = {patch_size_found}")
            processor.patch_size = patch_size_found
        else:
            # If not found or not an integer, fall back to the default and log a warning
            logging.warning(f"Could not reliably determine patch_size from image processor config. Defaulting processor.patch_size to {EXPECTED_PATCH_SIZE}.")
            processor.patch_size = EXPECTED_PATCH_SIZE
    else:
        # If it was already set correctly during loading, just log it
        logging.info(f"Processor 'patch_size' was already correctly set during loading: {processor.patch_size}.")
    # --- *** END: Ensure processor.patch_size is set *** ---


    # Log image processor info (useful for verification)
    if hasattr(processor, 'image_processor') and processor.image_processor is not None:
        img_proc = processor.image_processor
        img_proc_config = getattr(img_proc, 'config', {}) # Get config dict or empty dict
        logging.info(f"Image Processor Info: Type={type(img_proc)}, "
                     f"Size Config={img_proc_config.get('size', 'N/A')}, " # Get size from config
                     f"Resize Config={img_proc_config.get('do_resize', 'N/A')}, " # Get resize from config
                     f"Patch Size Config={img_proc_config.get('patch_size', 'N/A')}") # Get patch_size from config
    else:
         logging.warning("Processor does not have an 'image_processor' attribute or it is None.")

    logging.info("Processor loading and patch_size verification complete.")
    if processor.tokenizer and hasattr(processor.tokenizer, 'added_tokens_decoder'):
        if any(token.special for token in processor.tokenizer.added_tokens_decoder.values()):
            logging.info("Special tokens detected in tokenizer.")

except Exception as e:
    logging.critical(f"Fatal: Error loading processor or setting patch_size from {args.base_model_path}: {e}")
    traceback.print_exc()
    exit(1)


# --- Load Base Model using LlavaForConditionalGeneration ---
logging.info(f"Loading BASE model from: {args.base_model_path} using LlavaForConditionalGeneration")
try:
    base_model_params = {
        "low_cpu_mem_usage": True,
        "device_map": "auto"
    }
    if model_dtype:
        base_model_params["torch_dtype"] = model_dtype
    if args.load_in_4bit and quantization_config:
        base_model_params["quantization_config"] = quantization_config

    # *** Load the correct model class ***
    base_model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_path,
        **base_model_params
    )
    logging.info(f"Base model loaded. Type: {type(base_model)}")
    logging.info(f"Base model loaded. Device map (if used):\n{getattr(base_model, 'hf_device_map', 'Not using device_map explicitly for all layers / Not applicable')}")

    # Tokenizer/Embedding Resize Logic (Keep as is, but be mindful of the warning)
    if processor.tokenizer and hasattr(processor.tokenizer, 'added_tokens_decoder') and len(processor.tokenizer.added_tokens_decoder) > 0:
        current_model_vocab_size = None
        # Prefer text_config, then language_model_config, then direct config for vocab_size
        config_to_check = getattr(base_model.config, 'text_config', None)
        if config_to_check and hasattr(config_to_check, 'vocab_size'):
            current_model_vocab_size = config_to_check.vocab_size
        else:
            config_to_check = getattr(base_model.config, 'language_model_config', None)
            if config_to_check and hasattr(config_to_check, 'vocab_size'):
                current_model_vocab_size = config_to_check.vocab_size
            elif hasattr(base_model.config, 'vocab_size'):
                current_model_vocab_size = base_model.config.vocab_size

        new_tokenizer_size = len(processor.tokenizer)
        logging.info(f"Tokenizer size: {new_tokenizer_size}, Model's LM vocab size from config: {current_model_vocab_size}")

        if current_model_vocab_size is not None and new_tokenizer_size > current_model_vocab_size:
            logging.info(f"Resizing token embeddings from {current_model_vocab_size} to {new_tokenizer_size}")
            base_model.resize_token_embeddings(new_tokenizer_size)
            # Update vocab size in config AFTER resizing (important for merged models)
            if hasattr(base_model.config, 'text_config') and hasattr(base_model.config.text_config, 'vocab_size'):
                 base_model.config.text_config.vocab_size = new_tokenizer_size
            elif hasattr(base_model.config, 'language_model_config') and hasattr(base_model.config.language_model_config, 'vocab_size'):
                 base_model.config.language_model_config.vocab_size = new_tokenizer_size
            elif hasattr(base_model.config, 'vocab_size'):
                 base_model.config.vocab_size = new_tokenizer_size

        elif current_model_vocab_size is not None and new_tokenizer_size < current_model_vocab_size:
             logging.warning(f"Tokenizer size ({new_tokenizer_size}) is smaller than model's original vocab size ({current_model_vocab_size}). This is unusual. Not resizing downwards.")
        elif current_model_vocab_size is not None :
             logging.info("Tokenizer size matches model's LM vocab size. No embedding resize needed.")
        else:
            logging.warning("Could not determine model's original LM vocab_size from config. Skipping resize check.")

except Exception as e:
    logging.critical(f"Fatal: Error loading base model or resizing embeddings: {e}")
    traceback.print_exc()
    exit(1)


# --- Load and Apply PEFT LoRA Adapters ---
logging.info(f"Loading LoRA adapters: {args.adapter_repo}" + (f" (subfolder: {args.lora_subfolder})" if args.lora_subfolder else ""))
try:
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_repo,
        subfolder=args.lora_subfolder if args.lora_subfolder else None,
        is_trainable=False # Ensure model is in inference mode
    )
    logging.info("LoRA adapters loaded successfully.")
except Exception as e:
    logging.critical(f"Fatal: Error loading LoRA adapters from {args.adapter_repo}: {e}")
    traceback.print_exc()
    exit(1)

# --- Optional: Merge Adapters ---
if args.merge_adapters:
    logging.info("Merging LoRA adapters into base model...")
    try:
        if args.load_in_4bit:
             logging.warning("Merging adapters with a 4-bit quantized model. This will dequantize affected layers and significantly increase memory usage.")
        # Note: merge_and_unload() might require specific handling for device_map='auto' depending on transformers/peft versions.
        # It usually returns the base model instance.
        model = model.merge_and_unload()
        logging.info("Adapters merged successfully.")
        logging.info(f"Merged model device map (if available):\n{getattr(model, 'hf_device_map', 'N/A')}")
        logging.info(f"Merged model type: {type(model)}")
    except Exception as e:
        logging.error(f"Could not merge adapters: {e}. Using PEFT model (unmerged).")
        traceback.print_exc()

model.eval()
logging.info("Model ready for caption inference.")

# --- Define Prompt Template for Captioning ---
USER_PROMPT_INSTRUCTION = "¿Cuál es la descripción o el pie de foto de esta imagen médica?"
ASSISTANT_RESPONSE_PREFIX = ""
DEFAULT_IMAGE_TOKEN = "<image>"
try:
    image_token_str = getattr(processor, 'image_token', None)
    if image_token_str is None and hasattr(processor, 'tokenizer'):
        image_token_str = getattr(processor.tokenizer, 'image_token', None)
        if image_token_str is None:
             if DEFAULT_IMAGE_TOKEN in processor.tokenizer.get_vocab():
                 image_token_str = DEFAULT_IMAGE_TOKEN
             else:
                 logging.warning(f"Could not reliably determine image token. Defaulting to {DEFAULT_IMAGE_TOKEN}")
                 image_token_str = DEFAULT_IMAGE_TOKEN
    if image_token_str is None:
         image_token_str = DEFAULT_IMAGE_TOKEN
except Exception as e:
    logging.warning(f"Error determining image token: {e}. Defaulting to {DEFAULT_IMAGE_TOKEN}")
    image_token_str = DEFAULT_IMAGE_TOKEN

PROMPT_TEMPLATE = f"USER: {image_token_str}\n{USER_PROMPT_INSTRUCTION}\nASSISTANT:{ASSISTANT_RESPONSE_PREFIX}"
logging.info(f"Using Inference Prompt Template (ends exactly after ASSISTANT:):\n'{PROMPT_TEMPLATE}'")


# --- Image Processing and Caption Generation ---
image_files = sorted(glob.glob(os.path.join(args.image_dir, '*')))
valid_image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
logging.info(f"Found {len(valid_image_files)} images in {args.image_dir}")

if not valid_image_files:
    logging.critical(f"No valid image files found in {args.image_dir}. Exiting.")
    exit(1)

results = []
num_batches = (len(valid_image_files) + args.batch_size - 1) // args.batch_size

input_target_device = model.device
logging.info(f"Target device for input tensors: {input_target_device}")

expected_pixel_dtype = getattr(model, 'dtype', torch.float16)
if input_target_device.type == 'cpu':
    expected_pixel_dtype = torch.float32
if args.load_in_4bit and quantization_config:
    expected_pixel_dtype = torch.float16
logging.info(f"Expected pixel value dtype based on model/device: {expected_pixel_dtype}")


# ==============================================================================
# GENERATION LOOP (Using standard processor call, patch_size now fixed)
# ==============================================================================
for i in tqdm(range(num_batches), desc="Generating Captions"):
    batch_files = valid_image_files[i * args.batch_size : (i + 1) * args.batch_size]
    batch_images = []
    batch_ids = []

    for image_path in batch_files:
        try:
            image = Image.open(image_path).convert("RGB")
            batch_images.append(image)
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            batch_ids.append(image_id)
        except Exception as e:
            logging.warning(f"Skipping image {image_path} due to error opening/reading: {e}")
            continue

    if not batch_images:
        logging.warning(f"Batch {i+1}/{num_batches} is empty after skipping images.")
        continue

    current_batch_ids_str = ", ".join(batch_ids[:3]) + ("..." if len(batch_ids) > 3 else "")

    inputs = None
    try:
        # Use the standard processor call - processor.patch_size should now be set correctly
        # Added max_length as per recommendation
        inputs = processor(
            text=[PROMPT_TEMPLATE] * len(batch_images),
            images=batch_images,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=2048  # Explicitly set max_length for the processor
        ).to(input_target_device)

        # Ensure pixel_values dtype matches model expectation
        if 'pixel_values' in inputs and inputs['pixel_values'].dtype != expected_pixel_dtype:
            logging.debug(f"Batch {i+1}: Converting pixel_values dtype from {inputs['pixel_values'].dtype} to {expected_pixel_dtype}")
            inputs['pixel_values'] = inputs['pixel_values'].to(expected_pixel_dtype)

        logging.debug(f"Batch {i+1}: Inputs prepared by processor. Keys: {list(inputs.keys())}")
        if 'pixel_values' in inputs:
            logging.debug(f"Batch {i+1}: Pixel values shape: {inputs['pixel_values'].shape}, dtype: {inputs['pixel_values'].dtype}")
        if 'input_ids' in inputs:
            logging.debug(f"Batch {i+1}: Input IDs shape: {inputs['input_ids'].shape}")

    except Exception as e:
        logging.error(f"Error during processor call for batch {i+1} ({current_batch_ids_str}): {e}")
        traceback.print_exc() # Print full traceback for processor errors
        if inputs: del inputs
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        continue

    # Generate Captions
    output_ids = None
    generated_ids_only = None
    with torch.inference_mode():
        try:
            # --- INSERTED SNIPPET ---
            # Consolidate generation arguments
            gen_kwargs = {
                **generation_params,  # Your existing params like do_sample from args.generation_kwargs
                "max_new_tokens": args.max_new_tokens,
                "eos_token_id": processor.tokenizer.eos_token_id,
                "pad_token_id": processor.tokenizer.eos_token_id  # Explicitly set PAD to EOS
            }
            # --- END OF INSERTED SNIPPET ---

            output_ids = model.generate(
                **inputs,
                **gen_kwargs # Use the consolidated generation arguments
            )

            input_token_len = inputs['input_ids'].shape[1]
            generated_ids_only = output_ids[:, input_token_len:]
            generated_captions = processor.batch_decode(generated_ids_only, skip_special_tokens=True)

            logging.debug(f"Batch {i+1}: Successfully generated and decoded captions.")

        except Exception as e:
            logging.error(f"Error during model.generate() or decoding for batch {i+1} ({current_batch_ids_str}): {e}")
            traceback.print_exc() # Print full traceback for generation errors

    # Store results and Clean up
    if generated_ids_only is not None and generated_captions is not None:
        for img_id, caption_output in zip(batch_ids, generated_captions):
            final_caption = caption_output.strip()
            logging.debug(f"Image ID: {img_id}, Generated Caption: '{final_caption}'")
            #results.append({"ID": img_id, "Caption": final_caption})
            results.append({"ID": img_id, "Caption": final_caption})


    del inputs
    del output_ids
    del generated_ids_only
    if 'batch_images' in locals(): del batch_images # Clear image objects from memory too
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==============================================================================
# END OF GENERATION LOOP
# ==============================================================================


# --- Save Results ---
logging.info(f"Saving caption results to {args.output_csv}")
if not results:
    logging.warning("No results were generated. Saving an empty CSV with columns ID, Caption.")
    df = pd.DataFrame(columns=["ID", "Caption"])
else:
    df = pd.DataFrame(results)

output_dir = os.path.dirname(args.output_csv)
if output_dir and not os.path.exists(output_dir):
    try:
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create output directory {output_dir}. Error: {e}")
        args.output_csv = os.path.basename(args.output_csv)
        logging.warning(f"Attempting to save in current directory: ./{args.output_csv}")
try:
    if not df.empty:
        df = df[["ID", "Caption"]]
    df.to_csv(args.output_csv, index=False)
    logging.info(f"Successfully saved results to {args.output_csv}")
except Exception as e:
    logging.error(f"Failed to save results to {args.output_csv}. Error: {e}")
    traceback.print_exc()

logging.info("Caption generation process complete.")