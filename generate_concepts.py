# generate_concepts_v1_modified.py

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
import traceback
import re

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Funciones para el mapeo de CUIs ---
def load_cui_mapping(cui_mapping_file):
    """
    Carga el archivo de mapeo de términos a CUIs para la conversión posterior.
    El archivo CSV debe tener las columnas "CUI" y "Name".
    
    Args:
        cui_mapping_file: Ruta al archivo cui_names.csv
        
    Returns:
        Un diccionario donde las claves son términos en lenguaje natural (diferentes variantes)
        y los valores son los CUIs correspondientes.
    """
    logging.info(f"Cargando mapeo CUI desde: {cui_mapping_file}")
    
    try:
        # Cargar el archivo de mapeo
        df = pd.read_csv(cui_mapping_file)
        
        # Verificar columnas esperadas (Case-sensitive to match user's "CUI,Name")
        required_columns = ['CUI', 'Name'] 
        
        # Attempt to normalize column names if exact match fails
        if 'CUI' not in df.columns and 'cui' in df.columns:
            df.rename(columns={'cui': 'CUI'}, inplace=True)
            logging.info("Normalized CSV column 'cui' to 'CUI'")
        if 'Name' not in df.columns:
            if 'Name' in df.columns: # all lowercase
                df.rename(columns={'Name': 'Name'}, inplace=True)
                logging.info("Normalized CSV column 'Name' to 'Name'")
            elif 'Preferred_name' in df.columns: # Capitalized 'P'
                 df.rename(columns={'Preferred_name': 'Name'}, inplace=True)
                 logging.info("Normalized CSV column 'Preferred_name' to 'Name'")
            elif 'preferred_name' in df.columns: # all lowercase 'p' (original script)
                 df.rename(columns={'preferred_name': 'Name'}, inplace=True)
                 logging.info("Normalized CSV column 'preferred_name' to 'Name'")


        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"El archivo {cui_mapping_file} no contiene las columnas requeridas: {missing_columns}. Columnas encontradas: {df.columns.tolist()}")

        mapping_dict = {}
        
        for _, row in df.iterrows():
            cui = row['CUI']
            name = str(row['Name']).strip()
            
            if name and cui:
                mapping_dict[name] = cui
                mapping_dict[name.lower()] = cui
                clean_name = re.sub(r'[.,;:!?()\[\]{}]', '', name.lower())
                mapping_dict[clean_name] = cui
                
                for suffix in ["disorder", "disease", "syndrome", "condition"]:
                    if name.lower().endswith(f" {suffix}"):
                        clean_name_no_suffix = name.lower()[:-len(suffix)-1].strip()
                        if clean_name_no_suffix: # Ensure not empty after stripping
                           mapping_dict[clean_name_no_suffix] = cui
                
                # Si hay sinónimos, agregarlos también (asumiendo columna se llama 'synonyms')
                if 'synonyms' in df.columns and pd.notna(row['synonyms']):
                    synonyms_str = str(row['synonyms'])
                    synonyms_list = [s.strip() for s in synonyms_str.split(';') if s.strip()]
                    for synonym in synonyms_list:
                        if synonym: # Redundant check, but safe
                            mapping_dict[synonym] = cui
                            mapping_dict[synonym.lower()] = cui
                            clean_synonym = re.sub(r'[.,;:!?()\[\]{}]', '', synonym.lower())
                            mapping_dict[clean_synonym] = cui
        
        logging.info(f"Se cargaron {len(mapping_dict)} términos de mapeo desde {cui_mapping_file}")
        return mapping_dict
    
    except FileNotFoundError:
        logging.error(f"Error: Archivo de mapeo CUI no encontrado en {cui_mapping_file}")
        return {}
    except Exception as e:
        logging.error(f"Error cargando archivo de mapeo CUI ({cui_mapping_file}): {e}")
        traceback.print_exc()
        return {}

def convert_natural_to_cui(natural_terms_comma_separated, mapping_dict):
    """
    Convierte términos en lenguaje natural (separados por comas) a códigos CUI.
    
    Args:
        natural_terms_comma_separated: String con términos en lenguaje natural separados por comas.
        mapping_dict: Diccionario de mapeo términos -> CUIs.
        
    Returns:
        String con CUIs separados por punto y coma.
    """
    if not natural_terms_comma_separated or not isinstance(natural_terms_comma_separated, str) or not mapping_dict:
        return ""
    
    natural_terms_comma_separated = natural_terms_comma_separated.strip('"\'')
    terms_list = [term.strip() for term in natural_terms_comma_separated.split(',') if term.strip()]
    
    cui_list = []
    unmapped_terms = []
    
    for term in terms_list:
        # Term itself is already stripped and checked for emptiness by list comprehension
            
        term_variants = [
            term, # Original, as processed from split
            term.lower(), # Lowercase
            re.sub(r'[.,;:!?()\[\]{}]', '', term.lower()), # Lowercase, no punctuation
        ]
        
        # Add variants without common medical suffixes
        for suffix in ["disorder", "disease", "syndrome", "condition"]:
            if term.lower().endswith(f" {suffix}"):
                base_term = term.lower()[:-len(suffix)-1].strip()
                if base_term: # Ensure not empty
                    term_variants.append(base_term) # Already lowercased
                    term_variants.append(re.sub(r'[.,;:!?()\[\]{}]', '', base_term)) # Lowercased, no punctuation
        
        cui_found = None
        # More specific matches first (e.g., with punctuation, original case)
        # or iterate variants in a deliberate order of preference if needed.
        # Current order: Original, lower, clean_lower, suffix_removed, clean_suffix_removed
        for variant in term_variants: 
            if variant in mapping_dict:
                cui_found = mapping_dict[variant]
                break # Take the first match based on variant order
        
        if cui_found:
            cui_list.append(cui_found)
        else:
            unmapped_terms.append(term) # Append original term that was not mapped
    
    if unmapped_terms:
        if len(unmapped_terms) <= 5: # Log first few unmapped terms for easier debugging
            logging.warning(f"No se pudieron mapear los siguientes términos a CUIs: {', '.join(unmapped_terms)}")
        else:
            logging.warning(f"No se pudieron mapear {len(unmapped_terms)} términos a CUIs. Primeros 5: {', '.join(unmapped_terms[:5])}")
    
    return ";".join(cui_list)

# --- Configuration ---
parser = argparse.ArgumentParser(description="Generate medical concepts. First, it generates natural language concepts (Format A), then translates them to CUI codes (Format B).")

# Model Paths
parser.add_argument("--base_model_path", type=str, default="xtuner/llava-llama-3-8b-v1_1-transformers", help="Path or HuggingFace repo ID of the BASE Llava-Llama model.")
parser.add_argument("--adapter_repo", type=str, default="JoVal26/ja-med-clef-model", help="HuggingFace repo ID containing the PEFT LoRA adapters.")

# Data and Output Paths
parser.add_argument("--image_dir", type=str, default="./data/test/images", help="Directory containing images for concept extraction.")
parser.add_argument("--cui_mapping_file", type=str, default="./data/test/cui_names.csv", help="Path to the CUI mapping CSV file (must have 'CUI' and 'Name' columns).")
parser.add_argument(
    "--output_csv",
    type=str,
    default="submission.csv",
    help="Path to save the output concept submission CSV file.",
)
parser.add_argument(
    "--output_csv_natural",
    type=str,
    default="submission_natural.csv",
    help="Path to save the output concept submission CSV file.",
)


# Processing Parameters
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device ('cuda' or 'cpu').")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
parser.add_argument("--max_new_tokens", type=int, default=150, help="Max new tokens for generation.")
parser.add_argument("--merge_adapters", action='store_true', help="Merge LoRA adapters into base model.")
parser.add_argument("--load_in_4bit", action='store_true', help="Load base model in 4-bit.")
parser.add_argument("--lora_subfolder", type=str, default=None, help="Subfolder in adapter_repo for LoRA files.")
parser.add_argument("--generation_kwargs", type=str, default='{"do_sample": false}', help='JSON string for model.generate() kwargs.')
parser.add_argument("--auto_map_cuis", action='store_true', default=True, help="Enable translation to CUI codes for Format B. If false or mapping file is missing/invalid, Format B CSV will be empty or may not be fully populated.")

# Checkpointing
parser.add_argument("--checkpoint_interval", type=int, default=50, help="Save checkpoint every N batches (0 to disable). Default 50 batches.")
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_natural", help="Directory for checkpoint files (saving natural concepts).")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint file to resume from.")

args = parser.parse_args()

# --- Checkpoint Functions ---
def save_checkpoint(processed_data_list, completed_image_paths, checkpoint_file_path):
    checkpoint_parent_dir = os.path.dirname(checkpoint_file_path)
    if checkpoint_parent_dir and not os.path.exists(checkpoint_parent_dir):
        os.makedirs(checkpoint_parent_dir, exist_ok=True)
        
    checkpoint_content = {
        "processed_data": processed_data_list, 
        "completed_images": completed_image_paths,
        "args": vars(args), 
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    try:
        with open(checkpoint_file_path, 'w') as f:
            json.dump(checkpoint_content, f, indent=4)
        logging.info(f"Checkpoint saved to {checkpoint_file_path}")
        
        if processed_data_list:
            df_check = pd.DataFrame(processed_data_list)
            if not df_check.empty and "ID" in df_check.columns and "NaturalConcepts" in df_check.columns:
                df_check = df_check[["ID", "NaturalConcepts"]] 
                csv_checkpoint_path = checkpoint_file_path.replace('.json', '_natural_concepts.csv')
                df_check.to_csv(csv_checkpoint_path, index=False)
                logging.info(f"Natural concepts from checkpoint saved to {csv_checkpoint_path}")
            elif not df_check.empty:
                 logging.warning("Checkpoint DataFrame for CSV is missing 'ID' or 'NaturalConcepts' columns.")
    except Exception as e: 
        logging.error(f"Failed to save checkpoint to {checkpoint_file_path}: {e}")
        traceback.print_exc()


def load_checkpoint(checkpoint_file_path):
    try:
        with open(checkpoint_file_path, 'r') as f:
            checkpoint_content = json.load(f)
        
        loaded_data = checkpoint_content.get("processed_data", [])
        loaded_completed_images = checkpoint_content.get("completed_images", [])
        timestamp = checkpoint_content.get("timestamp", "unknown")
        
        logging.info(f"Loaded checkpoint from {timestamp} with {len(loaded_data)} processed data entries.")
        logging.info(f"Resuming from {len(loaded_completed_images)} already processed images.")
        
        return loaded_data, loaded_completed_images
    except FileNotFoundError:
        logging.error(f"Checkpoint file not found: {checkpoint_file_path}")
        return [], []
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from checkpoint file: {checkpoint_file_path}")
        return [], []
    except Exception as e:
        logging.error(f"Failed to load checkpoint from {checkpoint_file_path}: {e}")
        traceback.print_exc()
        return [], []

# --- Image File Initialization ---
image_files_all = []
if args.image_dir and os.path.isdir(args.image_dir):
    image_files_all = sorted(glob.glob(os.path.join(args.image_dir, '*')))
    valid_image_files_all = [f for f in image_files_all if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    logging.info(f"Found {len(valid_image_files_all)} valid images in {args.image_dir}")
else:
    logging.critical(f"Image directory '{args.image_dir}' not found or is not a directory. Exiting.")
    exit(1)


if not valid_image_files_all:
    logging.critical(f"No valid image files found in {args.image_dir}. Exiting.")
    exit(1)

all_natural_language_results = [] 
completed_image_file_paths = []

if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
    logging.info(f"Attempting to resume from checkpoint: {args.resume_from_checkpoint}")
    all_natural_language_results, completed_image_file_paths = load_checkpoint(args.resume_from_checkpoint)
    # Ensure completed_image_file_paths contains full paths if image_dir might change or be relative
    # For simplicity, assuming paths stored in checkpoint are directly comparable.
    images_to_process_now = [f for f in valid_image_files_all if f not in completed_image_file_paths]
    logging.info(f"After loading checkpoint, {len(images_to_process_now)} images remain to be processed.")
else:
    images_to_process_now = valid_image_files_all

num_batches_to_run = (len(images_to_process_now) + args.batch_size - 1) // args.batch_size

if args.checkpoint_interval > 0 and num_batches_to_run > 0 : 
    if not os.path.exists(args.checkpoint_dir):
      try:
          os.makedirs(args.checkpoint_dir, exist_ok=True)
          logging.info(f"Created checkpoint directory: {args.checkpoint_dir}")
      except OSError as e:
          logging.error(f"Could not create checkpoint directory {args.checkpoint_dir}. Checkpointing will likely fail. Error: {e}")
          args.checkpoint_interval = 0 # Disable checkpointing
    if args.checkpoint_interval > 0: # Re-check if disabling occurred
        logging.info(f"Checkpoints will be saved every {args.checkpoint_interval} batches to {args.checkpoint_dir}")


# --- CUI Mapping Setup ---
cui_mapping_dictionary = {}
if args.auto_map_cuis:
    cui_map_file_to_load = args.cui_mapping_file
    if not os.path.exists(cui_map_file_to_load):
        logging.warning(f"CUI mapping file not found at specified path: {cui_map_file_to_load}. Attempting to locate common alternatives...")
        # Determine base directory for relative paths, e.g., script's dir or image_dir's parent
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
        potential_paths = [
            os.path.join(script_dir, "cui_names.csv"), 
            os.path.join(script_dir, "data", "cui_names.csv"),
            os.path.join(os.path.dirname(args.image_dir), "cui_names.csv"), 
            os.path.join(os.path.dirname(os.path.dirname(args.image_dir)), "cui_names.csv") 
        ]
        found_path = None
        for p_path in potential_paths:
            if os.path.exists(p_path):
                cui_map_file_to_load = os.path.abspath(p_path)
                logging.info(f"Found CUI mapping file at alternative location: {cui_map_file_to_load}")
                found_path = cui_map_file_to_load
                break
        if not found_path:
            logging.error(f"Critical: CUI mapping file ('{args.cui_mapping_file}' or alternatives) not found. Format B (CUI conversion) will be skipped.")
            args.auto_map_cuis = False 
    
    if args.auto_map_cuis: 
        cui_mapping_dictionary = load_cui_mapping(cui_map_file_to_load)
        if not cui_mapping_dictionary:
            logging.error(f"Failed to load CUI mapping data from {cui_map_file_to_load}. Format B (CUI conversion) will be skipped.")
            args.auto_map_cuis = False
        else:
            logging.info(f"Successfully loaded {len(cui_mapping_dictionary)} CUI mappings from {cui_map_file_to_load}.")
else:
    logging.info("CUI mapping (Format B generation) is disabled by user via --auto_map_cuis=False.")

# --- Device Setup ---
cli_device_preference = torch.device(args.device)
logging.info(f"CLI device preference: {cli_device_preference}")
if cli_device_preference.type == 'cuda':
    if torch.cuda.is_available():
        logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("CUDA specified but not available. Switching to CPU.")
        cli_device_preference = torch.device("cpu")
        args.device = "cpu" # Update args to reflect actual device
elif cli_device_preference.type != 'cpu':
    logging.warning(f"Unsupported device '{args.device}' specified. Defaulting to CPU.")
    cli_device_preference = torch.device("cpu")
    args.device = "cpu"

# --- Parse Generation Kwargs ---
try:
    generation_params_from_arg = json.loads(args.generation_kwargs)
    logging.info(f"Using generation parameters from --generation_kwargs: {generation_params_from_arg}")
except json.JSONDecodeError as e:
    logging.error(f"Invalid JSON string for --generation_kwargs: {args.generation_kwargs}. Error: {e}")
    logging.warning("Falling back to default generation parameters: {'do_sample': False}")
    generation_params_from_arg = {"do_sample": False}

# --- Quantization Config ---
quantization_config = None
model_dtype = torch.float16 if cli_device_preference.type == 'cuda' and not args.load_in_4bit else torch.float32

if args.load_in_4bit:
    if cli_device_preference.type == 'cuda' and torch.cuda.is_available():
        try:
            logging.info("Setting up 4-bit quantization.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 
            )
            model_dtype = None # Handled by BitsAndBytesConfig
        except ImportError:
            logging.warning("BitsAndBytesConfig not found (bitsandbytes not installed or import issue). Proceeding without 4-bit.")
            args.load_in_4bit = False
            model_dtype = torch.float16 # Fallback for CUDA if 4-bit fails
        except Exception as e:
            logging.error(f"Error setting up 4-bit quantization: {e}. Proceeding without 4-bit.")
            args.load_in_4bit = False
            model_dtype = torch.float16 # Fallback
    else:
        logging.warning("4-bit loading requested but no CUDA GPU available or CUDA not working. Disabling 4-bit.")
        args.load_in_4bit = False
        model_dtype = torch.float32 # CPU or no CUDA, use float32

elif cli_device_preference.type == 'cpu': # Explicitly set float32 for CPU if not 4-bit
    model_dtype = torch.float32
    logging.info("Using CPU, ensuring model dtype is float32.")

# --- Load Processor (AutoProcessor will choose correctly for LLaVA v1.5) ---
logging.info(f"Loading processor using AutoProcessor from: {args.base_model_path}")
try:
    processor = AutoProcessor.from_pretrained(args.base_model_path)
    logging.info(f"Successfully loaded processor using AutoProcessor. Type: {type(processor)}")

    # Ensure processor.patch_size is set (critical for LlavaProcessor)
    EXPECTED_PATCH_SIZE = 14 # Standard for ViT-L/14 used in LLaVA v1.5

    if not hasattr(processor, 'patch_size') or processor.patch_size is None:
        logging.warning("Processor object attribute 'patch_size' is missing or None. Attempting to set it.")
        patch_size_found = None
        if hasattr(processor, 'image_processor') and processor.image_processor is not None:
            if hasattr(processor.image_processor, 'config') and processor.image_processor.config is not None:
                 patch_size_found = getattr(processor.image_processor.config, 'patch_size', None)
                 if patch_size_found:
                     logging.info(f"Found patch_size={patch_size_found} in image_processor.config.")
        
        if patch_size_found and isinstance(patch_size_found, int):
            logging.info(f"Setting processor.patch_size = {patch_size_found}")
            processor.patch_size = patch_size_found
        else:
            logging.warning(f"Could not reliably determine patch_size. Defaulting processor.patch_size to {EXPECTED_PATCH_SIZE}.")
            processor.patch_size = EXPECTED_PATCH_SIZE
    else:
        logging.info(f"Processor 'patch_size' was already set: {processor.patch_size}.")

    if hasattr(processor, 'image_processor') and processor.image_processor is not None:
        img_proc_conf = getattr(processor.image_processor, 'config', {})
        logging.info(f"Image Processor Info: Type={type(processor.image_processor)}, "
                     f"Size Config={img_proc_conf.get('size', 'N/A')}, "
                     f"Resize Config={img_proc_conf.get('do_resize', 'N/A')}, "
                     f"Patch Size Config={img_proc_conf.get('patch_size', 'N/A')}")
    else:
        logging.warning("Processor does not have an 'image_processor' attribute or it is None.")
    logging.info("Processor loading and patch_size verification complete.")

except Exception as e:
    logging.critical(f"Fatal: Error loading processor or setting patch_size from {args.base_model_path}: {e}")
    traceback.print_exc()
    exit(1)

# --- Load Base Model (Use LlavaForConditionalGeneration for v1.5) ---
logging.info(f"Loading BASE model from: {args.base_model_path} using LlavaForConditionalGeneration")
try:
    base_model_params = {
        "low_cpu_mem_usage": True,
        "device_map": "auto" # Recommended for handling large models and quantization
    }
    if model_dtype: # model_dtype will be None if 4-bit
        base_model_params["torch_dtype"] = model_dtype
    if args.load_in_4bit and quantization_config:
        base_model_params["quantization_config"] = quantization_config

    base_model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_path,
        **base_model_params
    )
    logging.info(f"Base model loaded. Type: {type(base_model)}")
    logging.info(f"Base model device map (if used):\n{getattr(base_model, 'hf_device_map', 'N/A')}")

    # Tokenizer/Embedding Resize Logic
    if processor.tokenizer and hasattr(processor.tokenizer, 'added_tokens_decoder') and len(processor.tokenizer.added_tokens_decoder) > 0:
        current_model_vocab_size = None
        config_to_check = getattr(base_model.config, 'text_config', getattr(base_model.config, 'language_model_config', base_model.config))
        if hasattr(config_to_check, 'vocab_size'):
            current_model_vocab_size = config_to_check.vocab_size
        
        new_tokenizer_size = len(processor.tokenizer)
        logging.info(f"Tokenizer size: {new_tokenizer_size}, Model's LM vocab size from config: {current_model_vocab_size}")

        if current_model_vocab_size is not None and new_tokenizer_size > current_model_vocab_size:
            logging.info(f"Resizing token embeddings from {current_model_vocab_size} to {new_tokenizer_size}")
            base_model.resize_token_embeddings(new_tokenizer_size)
            # Update vocab size in model's config
            if hasattr(config_to_check, 'vocab_size'):
                config_to_check.vocab_size = new_tokenizer_size
        elif current_model_vocab_size is not None and new_tokenizer_size < current_model_vocab_size:
             logging.warning(f"Tokenizer size ({new_tokenizer_size}) is smaller than model's LM vocab size ({current_model_vocab_size}). Not resizing downwards.")
        elif current_model_vocab_size is not None:
             logging.info("Tokenizer size matches model's LM vocab size. No embedding resize needed.")
        else:
            logging.warning("Could not determine model's original LM vocab_size from config. Skipping resize check.")
except Exception as e:
    logging.critical(f"Fatal: Error loading base model or resizing embeddings: {e}")
    traceback.print_exc()
    exit(1)

# --- Load and Apply PEFT LoRA Adapters ---
if args.adapter_repo:
    logging.info(f"Loading LoRA adapters from: {args.adapter_repo}" + (f" (subfolder: {args.lora_subfolder})" if args.lora_subfolder else ""))
    try:
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_repo,
            subfolder=args.lora_subfolder if args.lora_subfolder else None,
            is_trainable=False 
        )
        logging.info("LoRA adapters loaded successfully onto base model.")
    except Exception as e:
        logging.critical(f"Fatal: Error loading LoRA adapters from {args.adapter_repo}: {e}. Ensure adapter is compatible with the base model.")
        traceback.print_exc()
        exit(1)
else:
    logging.info("No adapter_repo specified. Using the base model as is.")
    model = base_model # Use the base_model directly

# --- Optional: Merge Adapters ---
if args.adapter_repo and args.merge_adapters: # Only merge if adapters were loaded
    if args.load_in_4bit:
        logging.warning("Merging adapters with a 4-bit loaded model will dequantize it, increasing VRAM. This is generally not recommended if 4-bit memory saving is the goal.")
    logging.info("Merging LoRA adapters into base model...")
    try:
        model = model.merge_and_unload() 
        logging.info(f"Adapters merged. Model type after merge: {type(model)}")
    except Exception as e:
        logging.error(f"Could not merge adapters: {e}. Proceeding with unmerged PEFT model (if adapters were loaded) or base model.")
        # If merge fails, 'model' still refers to the PeftModel or base_model.

model.eval() 
logging.info("Model ready for concept inference.")

# --- Define Inference Prompt ---
image_token_str = getattr(processor, 'image_token', '<image>') 
USER_PROMPT_INSTRUCTION = "Enumera los conceptos médicos clave (CUIs) observados o inferidos en esta imagen."
ASSISTANT_RESPONSE_PREFIX = "Los conceptos médicos clave son:" 
PROMPT_TEMPLATE = f"USER: {image_token_str}\n{USER_PROMPT_INSTRUCTION}\nASSISTANT:{ASSISTANT_RESPONSE_PREFIX}" 
logging.info(f"Using Inference Prompt Template (ends exactly after prefix):\n'{PROMPT_TEMPLATE}'")

# Determine target device for inputs (should be where the model's parameters are)
input_target_device = model.device 
logging.info(f"Target device for input tensors: {input_target_device}")

# Determine expected pixel dtype for the model's vision encoder
# Usually float16 for GPU (unless model.dtype is float32), float32 for CPU
expected_pixel_dtype = getattr(model, 'dtype', torch.float16) # Default to model's main dtype
if input_target_device.type == 'cpu':
    expected_pixel_dtype = torch.float32
# If 4-bit, vision inputs often still need to be float16 even if compute is different
if args.load_in_4bit and cli_device_preference.type == 'cuda': 
    expected_pixel_dtype = torch.float16
logging.info(f"Expected pixel value dtype for model inputs: {expected_pixel_dtype}")

# ==============================================================================
# GENERATION LOOP (Natural Language Concepts - Format A)
# ==============================================================================
if not images_to_process_now:
    logging.info("No new images to process (either all done from checkpoint or initial list was empty).")
else:
    logging.info(f"Starting generation loop for {len(images_to_process_now)} images in {num_batches_to_run} batches.")

processed_batches_count = 0
for i in tqdm(range(num_batches_to_run), desc="Generating Natural Language Concepts (Format A)", unit="batch"):
    batch_image_files = images_to_process_now[i * args.batch_size : (i + 1) * args.batch_size]
    if not batch_image_files: 
        logging.debug(f"Batch {i+1} is empty, skipping.")
        continue 

    batch_pil_images = []
    current_batch_image_ids = []

    for image_path in batch_image_files:
        try:
            image = Image.open(image_path).convert("RGB")
            batch_pil_images.append(image)
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            current_batch_image_ids.append(image_id)
        except FileNotFoundError:
            logging.warning(f"Image file not found: {image_path}. Skipping.")
        except Exception as e:
            logging.warning(f"Skipping image {image_path} due to error opening/processing: {e}")
            continue # Skip this image, but continue with batch
    
    if not batch_pil_images:
        logging.warning(f"Batch {i+1}/{num_batches_to_run} has no valid images to process after loading attempts.")
        completed_image_file_paths.extend(batch_image_files) 
        continue

    inputs = None
    try:
        # Max length for processor should consider both text and potential image tokens
        # LLaVA models typically have a context length like 2048 or 4096.
        # Processor's model_max_length attribute might be available.
        proc_max_len = getattr(processor, 'model_max_length', 2048)
        
        inputs = processor(
            text=[PROMPT_TEMPLATE] * len(batch_pil_images),
            images=batch_pil_images,
            return_tensors="pt",
            padding="longest", # Pad to the longest sequence in the current batch
            truncation=True,   # Truncate text if it exceeds model's capacity
            max_length=proc_max_len 
        )
        # Move all input tensors to the model's device(s)
        inputs = {k: v.to(input_target_device) if hasattr(v, 'to') else v for k, v in inputs.items()}


        if 'pixel_values' in inputs and inputs['pixel_values'].dtype != expected_pixel_dtype:
            logging.debug(f"Batch {i+1}: Converting pixel_values dtype from {inputs['pixel_values'].dtype} to {expected_pixel_dtype}")
            inputs['pixel_values'] = inputs['pixel_values'].to(expected_pixel_dtype)

    except Exception as e:
        ids_str = ", ".join(current_batch_image_ids[:3]) + ("..." if len(current_batch_image_ids) > 3 else "")
        logging.error(f"Error during processor step for batch {i+1} (IDs: {ids_str}): {e}")
        traceback.print_exc()
        completed_image_file_paths.extend(batch_image_files) 
        if inputs: del inputs
        if 'batch_pil_images' in locals(): del batch_pil_images
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        continue

    output_ids = None
    full_decoded_outputs = None
    with torch.inference_mode(): # Ensure no gradients are computed
        try:
            # Consolidate generation arguments
            # Use EOS as PAD for open-ended generation if no specific PAD token is set in tokenizer
            pad_token_id_to_use = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
            
            gen_config_dict = {
                **generation_params_from_arg, # User-defined (e.g., do_sample, num_beams)
                "max_new_tokens": args.max_new_tokens,
                "eos_token_id": processor.tokenizer.eos_token_id,
                "pad_token_id": pad_token_id_to_use 
            }
            
            output_ids = model.generate(**inputs, **gen_config_dict)
            
            # Decode the full output (prompt + generation)
            # skip_special_tokens=True is generally recommended for user-facing text.
            full_decoded_outputs = processor.batch_decode(output_ids, skip_special_tokens=True)
            logging.debug(f"Batch {i+1}: Successfully generated and decoded concepts.")

        except Exception as e:
            ids_str = ", ".join(current_batch_image_ids[:3]) + ("..." if len(current_batch_image_ids) > 3 else "")
            logging.error(f"Error during model.generate() or decoding for batch {i+1} (IDs: {ids_str}): {e}")
            traceback.print_exc()
            # Fall through to cleanup for this batch, results for this batch might be missing/empty

    if full_decoded_outputs:
        for img_id, full_output_text in zip(current_batch_image_ids, full_decoded_outputs):
            natural_concepts_raw_str = ""
            # Find the assistant's response part.
            # rfind is safer if prompt might contain the prefix.
            assistant_prefix_idx = full_output_text.rfind(ASSISTANT_RESPONSE_PREFIX)
            
            if assistant_prefix_idx != -1:
                natural_concepts_raw_str = full_output_text[assistant_prefix_idx + len(ASSISTANT_RESPONSE_PREFIX):].strip()
            else:
                # Fallback: if ASSISTANT_RESPONSE_PREFIX is not found, try to find "ASSISTANT:"
                # This can happen if the model varies slightly in its output format.
                assistant_turn_idx = full_output_text.rfind("ASSISTANT:") 
                if assistant_turn_idx != -1: # Found "ASSISTANT:"
                     natural_concepts_raw_str = full_output_text[assistant_turn_idx + len("ASSISTANT:"):].strip()
                     logging.debug(f"Img {img_id}: Used 'ASSISTANT:' fallback for prefix. Output: '{natural_concepts_raw_str[:70]}...'")
                else: # No clear prefix found
                    natural_concepts_raw_str = full_output_text.strip() 
                    logging.warning(f"Img {img_id}: Could not find assistant response prefix ('{ASSISTANT_RESPONSE_PREFIX}' or 'ASSISTANT:'). Using full output: '{natural_concepts_raw_str[:70]}...'")

            # Clean and prepare for Format A (semicolon-separated natural language terms)
            # Model might output comma-separated terms. Standardize to semicolon.
            format_a_natural_concepts = ""
            if natural_concepts_raw_str:
                terms = [term.strip() for term in natural_concepts_raw_str.split(',') if term.strip()]
                format_a_natural_concepts = ";".join(terms)
            
            all_natural_language_results.append({"ID": img_id, "NaturalConcepts": format_a_natural_concepts})
    else: # No decoded outputs (e.g., generation error before decoding)
        logging.warning(f"Batch {i+1}: No decoded outputs to process. Adding empty results for image IDs in this batch.")
        for img_id in current_batch_image_ids: # Add entries for all images attempted in batch
             all_natural_language_results.append({"ID": img_id, "NaturalConcepts": ""})


    completed_image_file_paths.extend(batch_image_files) # Mark these files as processed
    processed_batches_count +=1

    if args.checkpoint_interval > 0 and processed_batches_count % args.checkpoint_interval == 0:
        chkpt_filename = f"checkpoint_batch_{processed_batches_count}_of_{num_batches_to_run}.json"
        chkpt_path = os.path.join(args.checkpoint_dir, chkpt_filename)
        save_checkpoint(all_natural_language_results, completed_image_file_paths, chkpt_path)

    # Manual cleanup of tensors to free GPU memory
    del inputs, output_ids, full_decoded_outputs
    if 'batch_pil_images' in locals(): del batch_pil_images # list of PIL images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Final Checkpoint for Format A data (Natural Language Concepts) ---
if args.checkpoint_interval > 0 and num_batches_to_run > 0 and processed_batches_count > 0 : 
    final_chkpt_path = os.path.join(args.checkpoint_dir, "checkpoint_natural_concepts_final.json")
    save_checkpoint(all_natural_language_results, completed_image_file_paths, final_chkpt_path)
    logging.info(f"Final natural concepts checkpoint saved to {final_chkpt_path}")

# --- Ensure Output Directories Exist before saving final CSVs ---
for out_path_str in [args.output_csv_natural, args.output_csv]:
    if out_path_str: # Check if path is not None or empty
        out_dir = os.path.dirname(os.path.abspath(out_path_str)) # Use abspath for robustness
        if out_dir and not os.path.exists(out_dir): # Check if dirname is not empty (e.g. for relative paths in current dir)
            try:
                os.makedirs(out_dir, exist_ok=True)
                logging.info(f"Created output directory: {out_dir}")
            except OSError as e:
                logging.error(f"Could not create output directory {out_dir}. Error: {e}. Files might be saved in current directory if path was relative and dir creation failed.")


# --- Save Format A: Natural Language Concepts ---
logging.info(f"Preparing to save Format A (natural language concepts) to {args.output_csv_natural}")
if not all_natural_language_results:
    logging.warning("No natural language concepts were generated or loaded from checkpoint. Format A CSV will be empty.")
    # Create an empty DataFrame with the correct columns as specified by user for Format A
    df_format_a = pd.DataFrame(columns=["ID", "CUIs"]) 
else:
    df_format_a = pd.DataFrame(all_natural_language_results)
    # Ensure correct columns: "ID" and "NaturalConcepts" (which will be renamed to "CUIs")
    if "NaturalConcepts" not in df_format_a.columns and not df_format_a.empty:
        logging.warning("DataFrame for Format A is missing 'NaturalConcepts' column. Output may be incorrect.")
        df_format_a["NaturalConcepts"] = "" # Add empty if missing
    if "ID" not in df_format_a.columns and not df_format_a.empty:
        logging.warning("DataFrame for Format A is missing 'ID' column. Output may be incorrect.")
        df_format_a["ID"] = "" # Add empty if missing
    
    # Select and rename columns for Format A output file
    # The column containing natural language concepts is 'NaturalConcepts' internally,
    # but user's example for Format A output CSV shows this column named 'CUIs'.
    df_format_a = df_format_a.rename(columns={"NaturalConcepts": "CUIs"})
    df_format_a = df_format_a[["ID", "CUIs"]] 

try:
    df_format_a.to_csv(args.output_csv_natural, index=False)
    logging.info(f"Successfully saved Format A (natural language concepts) to {args.output_csv_natural}")
except Exception as e:
    logging.error(f"Failed to save Format A CSV to {args.output_csv_natural}. Error: {e}")
    traceback.print_exc()

# --- Translate to Format B (CUI Codes) and Save ---
if args.auto_map_cuis and cui_mapping_dictionary:
    logging.info(f"Translating natural concepts to CUIs for Format B, to be saved in {args.output_csv}")
    format_b_cui_results = []
    if not all_natural_language_results:
        logging.warning("No natural language concepts (Format A) available to translate for Format B.")
    else:
        for item in tqdm(all_natural_language_results, desc="Translating to CUIs (Format B)", unit="image"):
            # 'NaturalConcepts' column from all_natural_language_results contains semicolon-separated terms
            natural_concepts_str_semicolon_sep = item.get('NaturalConcepts', "") 
            
            # The convert_natural_to_cui function expects comma-separated terms
            natural_concepts_str_comma_sep = natural_concepts_str_semicolon_sep.replace(';', ',')
            
            cui_codes_str = convert_natural_to_cui(natural_concepts_str_comma_sep, cui_mapping_dictionary)
            format_b_cui_results.append({"ID": item['ID'], "CUIs": cui_codes_str})

    df_format_b = pd.DataFrame() # Initialize
    if not format_b_cui_results and all_natural_language_results: 
         logging.warning("Translation to CUIs resulted in no CUI codes for any item, or all items had no natural concepts. Format B CSV will be empty or reflect missing mappings.")
         df_format_b = pd.DataFrame(columns=["ID", "CUIs"]) 
    elif not format_b_cui_results and not all_natural_language_results: 
        logging.warning("No natural language data to translate. Format B CSV will be empty.")
        df_format_b = pd.DataFrame(columns=["ID", "CUIs"])
    elif format_b_cui_results : # Has CUI results
        df_format_b = pd.DataFrame(format_b_cui_results)
        # Ensure columns are "ID", "CUIs" for Format B output
        if "ID" not in df_format_b.columns and not df_format_b.empty: df_format_b["ID"] = ""
        if "CUIs" not in df_format_b.columns and not df_format_b.empty: df_format_b["CUIs"] = ""
        df_format_b = df_format_b[["ID", "CUIs"]] 
    else: # Should be covered by above, but as a fallback
        logging.info("No CUI results to save for Format B. Creating empty CSV.")
        df_format_b = pd.DataFrame(columns=["ID", "CUIs"])


    try:
        df_format_b.to_csv(args.output_csv, index=False)
        logging.info(f"Successfully saved Format B (CUI codes) to {args.output_csv}")
    except Exception as e:
        logging.error(f"Failed to save Format B CSV to {args.output_csv}. Error: {e}")
        traceback.print_exc()

elif not args.auto_map_cuis:
    logging.info(f"CUI mapping was disabled (--auto_map_cuis=False). Format B (CUI codes) will not be generated to {args.output_csv}.")
    logging.info(f"Creating an empty placeholder file for Format B at: {args.output_csv}")
    pd.DataFrame(columns=["ID", "CUIs"]).to_csv(args.output_csv, index=False)
elif not cui_mapping_dictionary: # auto_map_cuis was true, but dictionary failed to load
    logging.error(f"CUI mapping dictionary was not loaded (e.g., file not found or parse error). Format B (CUI codes) cannot be generated to {args.output_csv}.")
    logging.info(f"Creating an empty placeholder file for Format B at: {args.output_csv}")
    pd.DataFrame(columns=["ID", "CUIs"]).to_csv(args.output_csv, index=False)


logging.info("Concept generation and CUI translation process complete.")