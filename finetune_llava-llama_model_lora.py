# finetune_llava-llama_model_lora.py

import os
# Establecer TOKENIZERS_PARALLELISM
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random # Para la selección de plantillas de prompt

import pandas as pd
from PIL import Image
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import gc # Para la limpieza de memoria

# --- Configuración Esencial ---
model_id = "../llava-llama-3-8b-v1_1-transformers"
roco_base_path = "./data/ROCOv2-radiology-main/"
train_image_dir = os.path.join(roco_base_path, "source_dataset", "train", "processed_images")
valid_image_dir = os.path.join(roco_base_path, "source_dataset", "valid", "processed_images")
IMAGE_EXTENSION = "jpeg"

train_captions_path = os.path.join(roco_base_path, "source_dataset", "train_captions.csv")
train_concepts_path = os.path.join(roco_base_path, "source_dataset", "train_concepts.csv")
validation_captions_path = os.path.join(roco_base_path, "source_dataset", "valid_captions.csv")
validation_concepts_path = os.path.join(roco_base_path, "source_dataset", "valid_concepts.csv")
cui_mapping_path = os.path.join(roco_base_path, "source_dataset", "cui_mapping.csv")

output_dir = "./models/llava_llama3_8b_rocov2_finetuned_full" 
logging_dir = "./logs_llava_rocov2_finetuned_full"

RANDOM_STATE = 42
MAX_LENGTH = 1024 
NUM_TRAIN_EPOCHS = 1  
LEARNING_RATE = 5e-5  
PER_DEVICE_TRAIN_BATCH_SIZE = 4 
GRADIENT_ACCUMULATION_STEPS = 8 
LORA_R = 16; LORA_ALPHA = 32; LORA_DROPOUT = 0.05
EVAL_SAVE_STEPS = 250 
LOGGING_STEPS = 50    

# --- Verificar GPU ---
if not torch.cuda.is_available(): raise SystemError("CUDA no disponible.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
if torch.cuda.is_available():
    try:
        print(f"Nombre de GPU: {torch.cuda.get_device_name(0)}")
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Memoria GPU Total: {total_mem_gb:.2f} GB")
    except AssertionError: print("Advertencia: No se pudo obtener info de GPU.")

def load_and_process_split_metadata(split_name, captions_csv_path, concepts_csv_path, cui_to_name_dictionary_map):
    print(f"Cargando y preprocesando metadatos para la división: {split_name}...")
    try:
        for p in [captions_csv_path, concepts_csv_path]:
            if not os.path.exists(p): raise FileNotFoundError(f"Archivo no encontrado para {split_name}: {p}")
        df_captions_split = pd.read_csv(captions_csv_path); df_concepts_split = pd.read_csv(concepts_csv_path)
        if 'ID' not in df_captions_split.columns or 'Caption' not in df_captions_split.columns: raise ValueError("Captions CSV sin 'ID' o 'Caption'.")
        if 'ID' not in df_concepts_split.columns or 'CUIs' not in df_concepts_split.columns: raise ValueError("Concepts CSV sin 'ID' o 'CUIs'.")
        merged_df_split = pd.merge(df_captions_split, df_concepts_split, on="ID", how="inner")
        if merged_df_split.empty: print(f"ADVERTENCIA: Fusión para {split_name} vacía."); return pd.DataFrame(columns=['image_id', 'original_caption', 'CUIs', 'CUI_Names_List'])
        def get_cui_names_local(cui_string):
            if pd.isna(cui_string) or not isinstance(cui_string, str) or cui_string.strip() == "": return []
            return [cui_to_name_dictionary_map.get(cui.strip(), f"Desconocido({cui.strip()})") for cui in str(cui_string).split(';') if cui.strip()]
        merged_df_split['CUI_Names_List'] = merged_df_split['CUIs'].apply(get_cui_names_local)
        merged_df_split.rename(columns={'ID': 'image_id', 'Caption': 'original_caption'}, inplace=True)
        merged_df_split['original_caption'] = merged_df_split['original_caption'].fillna('').astype(str)
        return merged_df_split
    except Exception as e: print(f"Error en load_and_process_split_metadata ({split_name}): {e}"); import traceback; traceback.print_exc(); exit(1)

print("Cargando mapeo CUI...")
try:
    if not os.path.exists(cui_mapping_path): raise FileNotFoundError(f"Mapeo CUI no encontrado: {cui_mapping_path}")
    df_cui_map = pd.read_csv(cui_mapping_path)
    if 'CUI' not in df_cui_map.columns or 'Canonical name' not in df_cui_map.columns: raise ValueError("Mapeo CUI sin 'CUI' o 'Canonical name'.")
    cui_to_name_map_dict = pd.Series(df_cui_map['Canonical name'].values, index=df_cui_map['CUI'].astype(str)).to_dict()
    print("Mapeo CUI cargado.")
except Exception as e: print(f"Error Crítico al cargar cui_mapping.csv: {e}"); exit(1)

print(f"Cargando procesador para el modelo LLaVA: {model_id}")
try:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(processor, 'patch_size') or processor.patch_size is None: processor.patch_size = 14
    if hasattr(processor, 'image_processor') and processor.image_processor is not None:
         if not hasattr(processor.image_processor, 'patch_size') or processor.image_processor.patch_size is None:
            if hasattr(processor.image_processor, 'config') and hasattr(processor.image_processor.config, 'patch_size'): processor.image_processor.config.patch_size = 14
            else:
                try: processor.image_processor.patch_size = 14
                except AttributeError: print("ADVERTENCIA: No se pudo establecer patch_size en image_processor.")
    
    if processor.tokenizer.pad_token is None:
        print("WARN: tokenizer.pad_token es None. Estableciendo a eos_token.")
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    print(f"Procesador cargado. Pad token ID: {processor.tokenizer.pad_token_id}, Pad token: '{processor.tokenizer.pad_token}'")
    print(f"  EOS token ID: {processor.tokenizer.eos_token_id}, EOS token: '{processor.tokenizer.eos_token}'")
except Exception as e: print(f"Error crítico al cargar el procesador: {e}"); import traceback; traceback.print_exc(); exit(1)

image_token_for_prompt = getattr(processor, 'image_token', getattr(processor.tokenizer, 'image_token', '<image>'))
eos_token_str = processor.tokenizer.eos_token if processor.tokenizer.eos_token is not None else ""

# --- 🔑 CAMBIO: Restaurar create_full_prompt_column a la lógica original con datos reales ---
def create_full_prompt_column(df, tokenizer_image_token_str, tokenizer_eos_token_str):
    if df.empty: df['prompt'] = pd.Series(dtype=str); return df
    prompts_list = []
    for _, row in df.iterrows():
        caption = str(row['original_caption']).strip()
        cui_names_list = row['CUI_Names_List']
        if cui_names_list and isinstance(cui_names_list, list) and len(cui_names_list) > 0:
            cui_names_str = ', '.join(cui_names_list)
        else:
            cui_names_str = "ningún concepto médico específico fue identificado"
        
        prompt_templates = [
            {"user": f"{tokenizer_image_token_str}\n¿Cuál es la descripción o el pie de foto de esta imagen médica?", 
             "assistant": f"{caption}{tokenizer_eos_token_str}"},
            {"user": f"{tokenizer_image_token_str}\nEnumera los conceptos médicos clave (CUIs) observados o inferidos en esta imagen.", 
             "assistant": f"Los conceptos médicos clave son: {cui_names_str}{tokenizer_eos_token_str}"},
            {"user": f"{tokenizer_image_token_str}\nDescribe esta imagen médica en detalle y lista los conceptos médicos clave.", 
             "assistant": f"Descripción: {caption}\nConceptos médicos clave: {cui_names_str}{tokenizer_eos_token_str}"}
        ]
        if cui_names_list and isinstance(cui_names_list, list) and len(cui_names_list) > 0:
            specific_concept_to_ask_positive = random.choice(cui_names_list)
            assistant_for_specific_concept = f"Sí, el concepto '{specific_concept_to_ask_positive}' es relevante. {f'Otros conceptos: {cui_names_str}.' if len(cui_names_list) > 1 else ''}".strip()
            prompt_templates.append({
                "user": f"{tokenizer_image_token_str}\nCon respecto a esta imagen, ¿se identifica el concepto '{specific_concept_to_ask_positive}'?", 
                "assistant": f"{assistant_for_specific_concept}{tokenizer_eos_token_str}"
            })
            
            chosen_concept = random.choice(cui_names_list)
            assistant_for_conditional_desc = f"Con la presencia de '{chosen_concept}', la descripción es: {caption}. Otros conceptos relevantes son: {cui_names_str}."
            prompt_templates.append({
                "user": f"{tokenizer_image_token_str}\nConsiderando '{chosen_concept}', ¿cuál es la descripción detallada de la imagen?", 
                "assistant": f"{assistant_for_conditional_desc}{tokenizer_eos_token_str}"
            })
            
        selected_template = random.choice(prompt_templates)
        # El tokenizer_eos_token_str ya está en selected_template['assistant']
        full_prompt_for_model = f"USER: {selected_template['user']}\nASSISTANT: {selected_template['assistant']}"
        prompts_list.append(full_prompt_for_model)
        
    df['prompt'] = prompts_list
    return df

train_df = load_and_process_split_metadata("train", train_captions_path, train_concepts_path, cui_to_name_map_dict)
if train_df.empty: raise ValueError("Train_df vacío.")
eval_df = load_and_process_split_metadata("validation", validation_captions_path, validation_concepts_path, cui_to_name_map_dict)

print("Creando la columna 'prompt' para los dataframes (con datos reales)...")
train_df = create_full_prompt_column(train_df, image_token_for_prompt, eos_token_str)
if not eval_df.empty: eval_df = create_full_prompt_column(eval_df, image_token_for_prompt, eos_token_str)
if not train_df.empty: print(f"Ejemplo de prompt generado (train_df):\n{train_df['prompt'].iloc[0]}")
print(f"\nResumen de Carga de Datos: Tr: {len(train_df)}, Val: {len(eval_df)}")

print(f"Cargando modelo base LLaVA: {model_id}...")
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
model.config.use_cache = False
if processor.tokenizer.pad_token_id is not None:
    model.config.pad_token_id = processor.tokenizer.pad_token_id

if not hasattr(processor.tokenizer, "image_token") or processor.tokenizer.image_token is None: processor.tokenizer.image_token = "<image>"
if not hasattr(processor, "image_token") or processor.image_token is None: processor.image_token = processor.tokenizer.image_token
current_image_token_str = getattr(processor.tokenizer, 'image_token', '<image>')
if current_image_token_str not in processor.tokenizer.get_vocab():
    added_tokens = processor.tokenizer.add_special_tokens({"additional_special_tokens": [current_image_token_str]})
    if added_tokens > 0: model.resize_token_embeddings(len(processor.tokenizer))

def preprocess_data(single_example, current_image_dir):
    image_id = single_example.get("image_id")
    prompt_text = single_example.get("prompt") 
    if not image_id or not prompt_text: return {} 
    image_path = os.path.join(current_image_dir, f"{image_id}.{IMAGE_EXTENSION}")
    try:
        pil_image = Image.open(image_path).convert("RGB")
        tokenized_single = processor.tokenizer(text=[prompt_text], return_tensors="pt", padding="longest", truncation=True, max_length=MAX_LENGTH)
        image_single = processor.image_processor(images=[pil_image], return_tensors="pt")
        if not (isinstance(tokenized_single.get("input_ids"), torch.Tensor) and
                isinstance(tokenized_single.get("attention_mask"), torch.Tensor) and 
                isinstance(image_single.get("pixel_values"), torch.Tensor)): return {}
    except Exception: return {}
    return {
        "input_ids": tokenized_single["input_ids"],
        "attention_mask": tokenized_single["attention_mask"],
        "pixel_values": image_single["pixel_values"]
    }

hf_train_dataset_full = Dataset.from_pandas(train_df)
n_total = len(hf_train_dataset_full)
if n_total < 3: chunks_dfs = [hf_train_dataset_full]
else:
    n_chunk1 = n_total // 3; n_chunk2 = n_total // 3
    train_splits_1 = hf_train_dataset_full.train_test_split(train_size=n_chunk1, seed=RANDOM_STATE, shuffle=True)
    chunk1_ds = train_splits_1["train"]; rest_1_ds = train_splits_1["test"]
    if len(rest_1_ds) < 2: chunks_dfs = [chunk1_ds, rest_1_ds] if len(rest_1_ds)>0 else [chunk1_ds]
    else:
        n_rest_1 = len(rest_1_ds)
        train_splits_2 = rest_1_ds.train_test_split(train_size=min(n_chunk2, n_rest_1 // 2 if n_rest_1 > 1 else 1), seed=RANDOM_STATE, shuffle=True)
        chunk2_ds = train_splits_2["train"]; chunk3_ds = train_splits_2["test"]
        chunks_dfs = [chunk1_ds, chunk2_ds]; 
        if len(chunk3_ds) > 0: chunks_dfs.append(chunk3_ds)

print(f"Número de chunks a procesar para entrenamiento: {len(chunks_dfs)}")
processed_chunk_paths = []
for i, chunk_dataset_to_process in enumerate(chunks_dfs):
    chunk_name = f"chunk{i+1}"; save_path = f"./tmp_processed_{chunk_name}"; processed_chunk_paths.append(save_path)
    print(f"\n🔹 Procesando {chunk_name} ({len(chunk_dataset_to_process)} muestras)...")
    processed = chunk_dataset_to_process.map(
        preprocess_data, fn_kwargs={"current_image_dir": train_image_dir}, 
        batched=False, 
        remove_columns=[col for col in chunk_dataset_to_process.column_names if col not in ['input_ids', 'attention_mask', 'pixel_values']],
        load_from_cache_file=False, num_proc=1 
    )
    processed.save_to_disk(save_path); print(f"✅ {chunk_name} guardado en {save_path}"); del processed; gc.collect()

print("\n🔁 Cargando chunks procesados desde disco y concatenando...")
loaded_chunks = [load_from_disk(path) for path in processed_chunk_paths if os.path.exists(path)]
loaded_chunks = [chunk for chunk in loaded_chunks if len(chunk) > 0]
if not loaded_chunks: raise ValueError("Todos los chunks de entrenamiento vacíos.")
train_dataset = concatenate_datasets(loaded_chunks)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "pixel_values"])
print(f"✅ Dataset final de entrenamiento concatenado: {len(train_dataset)} muestras. Formato: {train_dataset.format['type']}")

if eval_df.empty: 
    processed_eval_dataset = None; print("Dataset de validación vacío.")
else:
    print("\n🔹 Procesando dataset de validación...")
    hf_eval_dataset = Dataset.from_pandas(eval_df)
    processed_eval_dataset = hf_eval_dataset.map(
        preprocess_data, fn_kwargs={"current_image_dir": valid_image_dir}, 
        batched=False, 
        remove_columns=[col for col in hf_eval_dataset.column_names if col not in ['input_ids', 'attention_mask', 'pixel_values']],
        load_from_cache_file=False, num_proc=1
    )
    if len(processed_eval_dataset) == 0:
        processed_eval_dataset = None; print("ADVERTENCIA: Dataset de validación vacío post-procesamiento.")
    else:
        processed_eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "pixel_values"])
        print(f"✅ Dataset de validación procesado: {len(processed_eval_dataset)} muestras. Formato: {processed_eval_dataset.format['type']}")

target_modules_base_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
if hasattr(model, 'language_model') and hasattr(model.language_model, 'config') and hasattr(model.language_model.config, 'num_hidden_layers'):
    num_layers = model.language_model.config.num_hidden_layers
else: num_layers = 32; print("ADVERTENCIA: num_hidden_layers no encontrado, usando 32.")
lora_target_modules = []
for i in range(num_layers):
    for name in target_modules_base_names:
        lora_target_modules.append(f"language_model.model.layers.{i}.self_attn.{name}")
        lora_target_modules.append(f"language_model.model.layers.{i}.mlp.{name}")
lora_target_modules.extend(["multi_modal_projector.linear_1", "multi_modal_projector.linear_2"])
existing_target_modules = []
for mod_name in lora_target_modules:
    try: 
        parent = model; parts = mod_name.split('.');
        for part in parts: parent = getattr(parent, part)
        existing_target_modules.append(mod_name)
    except AttributeError: pass 
target_modules = list(set(existing_target_modules))
print(f"Módulos LoRA objetivo: {target_modules}")
if not target_modules: print("ADVERTENCIA: No se encontraron módulos LoRA objetivo.")

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", target_modules=target_modules)
model = get_peft_model(model, peft_config)
print("Modelo adaptado con PEFT (LoRA)."); model.print_trainable_parameters()

# --- 🔑 TrainingArguments Ajustados para Entrenamiento Completo ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, # 4
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, # 8 (Batch efectivo 32)
    num_train_epochs=NUM_TRAIN_EPOCHS, # 3
    learning_rate=LEARNING_RATE, # 5e-5 (mantenido)
    
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),

    evaluation_strategy="steps" if processed_eval_dataset and len(processed_eval_dataset)>0 else "no",
    eval_steps=EVAL_SAVE_STEPS, # 250
    logging_strategy="steps", 
    logging_steps=LOGGING_STEPS, # 50
    save_strategy="steps", 
    save_steps=EVAL_SAVE_STEPS, # 250
    save_total_limit=2,
    
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    load_best_model_at_end=True, # 🔑 Reactivado
    metric_for_best_model="loss", # 🔑 Reactivado (asumiendo que eval_loss se reportará)
    
    gradient_checkpointing=True, 
    gradient_checkpointing_kwargs={'use_reentrant': False}, 
    dataloader_num_workers=4, # 🔑 Aumentado
    dataloader_pin_memory=True, 
    report_to=["tensorboard"], 
    seed=RANDOM_STATE, 
    logging_dir=logging_dir,
)

# --- Collator Personalizado con Creación Explícita de Labels ---
def multimodal_data_collator(features):
    input_ids_list = []
    attention_mask_list = []
    pixel_values_list = []

    for i, f_dict in enumerate(features):
        current_input_ids = f_dict.get("input_ids") 
        current_attention_mask = f_dict.get("attention_mask") 
        current_pixel_values = f_dict.get("pixel_values") 

        if not (isinstance(current_input_ids, torch.Tensor) and current_input_ids.ndim == 2 and current_input_ids.shape[0] == 1 and
                  isinstance(current_attention_mask, torch.Tensor) and current_attention_mask.ndim == 2 and current_attention_mask.shape[0] == 1 and
                  isinstance(current_pixel_values, torch.Tensor) and current_pixel_values.ndim == 4 and current_pixel_values.shape[0] == 1):
            continue
        
        input_ids_list.append(current_input_ids.squeeze(0)) 
        attention_mask_list.append(current_attention_mask.squeeze(0)) 
        pixel_values_list.append(current_pixel_values) 

    if not input_ids_list: 
        return {} 

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
        if pad_token_id is None: raise ValueError("pad_token_id y eos_token_id son None.")

    batch_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    batch_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    try:
        batch_pixel_values = torch.cat(pixel_values_list, dim=0) 
    except Exception as e_cat:
        raise e_cat 

    labels = batch_input_ids.clone()
    
    assistant_marker_str = "ASSISTANT:" # Marcador textual
    # Tokenizar el marcador SIN special tokens para obtener sus IDs puros
    assistant_marker_ids = processor.tokenizer.encode(assistant_marker_str, add_special_tokens=False)
    marker_tuple = tuple(assistant_marker_ids)
    
    for i in range(labels.shape[0]): 
        sequence_ids = batch_input_ids[i].tolist()
        seq_tuple = tuple(sequence_ids)
        marker_start_idx = -1
        # Buscar la secuencia de tokens del marcador
        for k in range(len(seq_tuple) - len(marker_tuple) + 1):
            if seq_tuple[k:k+len(marker_tuple)] == marker_tuple:
                marker_start_idx = k
                break
        
        if marker_start_idx != -1:
            label_mask_until_idx = marker_start_idx + len(assistant_marker_ids)
            labels[i, :label_mask_until_idx] = -100
        else:
            # print(f"WARN COLLATOR: Marcador 'ASSISTANT:' no encontrado en secuencia {i}. Enmascarando toda la etiqueta.")
            labels[i, :] = -100 
            
        labels[i][batch_input_ids[i] == pad_token_id] = -100
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "pixel_values": batch_pixel_values,
        "labels": labels 
    }

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=processed_eval_dataset, data_collator=multimodal_data_collator, tokenizer=processor.tokenizer)

print(f"Iniciando fine-tuning con LLaVA ({model_id}) en ROCOv2 para {NUM_TRAIN_EPOCHS} épocas...")
try:
    if len(train_dataset) > 0 :
        train_result = trainer.train()
        trainer.save_model() 
        if hasattr(train_result, 'metrics'): trainer.log_metrics("train", train_result.metrics)
        if hasattr(train_result, 'metrics'): trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
    else: print("ADVERTENCIA: El dataset de entrenamiento está vacío. Saltando el entrenamiento.")
except Exception as e_train: print(f"Error durante el entrenamiento: {e_train}"); import traceback; traceback.print_exc()

if processed_eval_dataset and len(processed_eval_dataset) > 0:
    print("\nRealizando evaluación final en el conjunto de validación...")
    try:
        metrics = trainer.evaluate()
        if "eval_loss" in metrics:
            trainer.log_metrics("eval", metrics); trainer.save_metrics("eval", metrics)
        else:
            print(f"WARN: 'eval_loss' no encontrado en las métricas de evaluación. Métricas obtenidas: {metrics}")
    except Exception as e_eval: print(f"Error durante la evaluación final: {e_eval}"); import traceback; traceback.print_exc()
else: print("\nDataset de evaluación vacío o no disponible. Saltando evaluación final.")

print("\nGuardando el modelo adaptador PEFT final...")
final_adapter_path = os.path.join(output_dir, "final_lora_adapter_explicit")
try: model.save_pretrained(final_adapter_path); print(f"Adaptador guardado en: {final_adapter_path}")
except Exception as e: print(f"Error al guardar adaptador: {e}")
print("\nGuardando el procesador...")
final_processor_path = os.path.join(output_dir, "final_processor")
try: processor.save_pretrained(final_processor_path); print(f"Procesador guardado en: {final_processor_path}")
except Exception as e: print(f"Error al guardar procesador: {e}")
print("\n¡Proceso completado!")
print(f"Artefactos del modelo en: {output_dir}")
print(f"Logs de TensorBoard en: {logging_dir}")
