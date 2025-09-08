# clef_jaimage_2025

## Overview

JJ-VMed is an experimental multimodal computational framework developed by **Jaimage** for the **ImageCLEFmed Caption Lab at CLEF 2025**. This repository implements automated **concept detection** and **caption generation** for medical imaging using vision-language models.

> **Note**: This repository focuses specifically on concept detection and caption generation tasks. Explainability analysis components are maintained in a separate repository.

## Framework Architecture

### Core Model
- **Base Model**: LLaVA-LLaMA 3 8B (`xtuner/llava-llama-3-8b-v1_1-transformers`)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) via PEFT
- **Language Enhancement**: Spanish prompting for cross-linguistic robustness
- **Memory Optimization**: 4-bit quantization support via BitsAndBytesConfig

### Four-Phase Methodology

1. **Preprocessing**: Image and text preprocessing with automatic resizing and normalization
2. **LoRA Fine-tuning**: Parameter-efficient adaptation using PEFT adapters
3. **Generation**: Batch inference with configurable generation parameters
4. **Post-processing**: CUI mapping, validation, and format standardization

## Features

### 🏥 Medical Concept Detection
- **Natural Language Extraction**: Identifies medical concepts in human-readable terms
- **CUI Code Mapping**: Automatic conversion to UMLS Concept Unique Identifiers
- **Dual Output Formats**: 
  - Format A: Natural language concepts (semicolon-separated)
  - Format B: CUI codes (semicolon-separated)
- **Robust Mapping**: Handles synonyms, variations, and medical terminology normalization

### 📝 Medical Caption Generation
- **Spanish Prompting**: Enhanced multilingual caption generation
- **Contextual Understanding**: Medical domain-specific fine-tuning
- **Batch Processing**: Efficient inference with configurable batch sizes
- **Quality Validation**: Post-processing for caption consistency

### ⚡ Technical Capabilities
- **Memory Efficient**: 4-bit quantization for reduced VRAM usage
- **Scalable Processing**: Configurable batch sizes and checkpointing
- **Device Flexibility**: CUDA/CPU support with automatic device mapping
- **Resume Functionality**: Checkpoint-based processing for large datasets
- **Error Handling**: Robust error recovery and logging

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/jj-vmed.git
cd jj-vmed

# Install dependencies
pip install torch torchvision transformers
pip install peft accelerate bitsandbytes
pip install pillow pandas tqdm

# For 4-bit quantization (optional)
pip install bitsandbytes
```

## Quick Start

### Concept Detection

```bash
python generate_concepts.py \
    --image_dir ./data/test/images \
    --output_csv ./results/concepts.csv \
    --output_csv_natural ./results/concepts_natural.csv \
    --cui_mapping_file ./data/cui_names.csv \
    --batch_size 4 \
    --max_new_tokens 150
```

### Caption Generation

```bash
python generate_captions.py \
    --image_dir ./data/test/images \
    --output_csv ./results/captions.csv \
    --batch_size 4 \
    --max_new_tokens 100
```

## Configuration Options

### Model Configuration
- `--base_model_path`: Base LLaVA model path or HuggingFace ID
- `--adapter_repo`: PEFT LoRA adapter repository ID
- `--merge_adapters`: Merge adapters into base model
- `--load_in_4bit`: Enable 4-bit quantization

### Processing Configuration
- `--batch_size`: Inference batch size (default: 4)
- `--max_new_tokens`: Maximum tokens to generate
- `--device`: Target device (cuda/cpu)
- `--generation_kwargs`: JSON string for generation parameters

### Advanced Features
- `--checkpoint_interval`: Automatic checkpointing frequency
- `--resume_from_checkpoint`: Resume from saved checkpoint
- `--auto_map_cuis`: Enable/disable CUI code mapping

## Output Formats

### Concept Detection
```csv
ID,CUIs
image_001,"pneumonia;chest x-ray;lung opacity"
image_002,"C0032285;C0039985;C0034067"
```

### Caption Generation
```csv
ID,Caption
image_001,"Radiografía de tórax que muestra opacidad pulmonar compatible con neumonía"
image_002,"TC de abdomen con contraste que revela hepatomegalia"
```

## Model Performance

### Technical Specifications
- **Parameters**: 8B (base model)
- **Context Length**: 2048 tokens
- **Image Resolution**: 336×336 (ViT-L/14)
- **Patch Size**: 14×14
- **Memory Usage**: ~16GB (FP16) / ~8GB (4-bit)

### Processing Capabilities
- **Batch Processing**: Up to 8 images simultaneously (depending on VRAM)
- **Throughput**: ~2-4 images/second (RTX 4090)
- **Languages**: Primary Spanish, fallback English

## Challenge Context

This implementation was developed for the **ImageCLEFmed Caption Lab at CLEF 2025**, addressing:

- **Task 1**: Medical concept detection and CUI mapping
- **Task 2**: Medical image caption generation
- **Domain**: Radiology, pathology, and clinical imaging
- **Languages**: Spanish (primary), English (secondary)

## Model Weights

The fine-tuned LoRA adapters are available on HuggingFace:
- **Repository**: `JoVal26/ja-med-clef-model`
- **Base Model**: `xtuner/llava-llama-3-8b-v1_1-transformers`

## Citation

```bibtex
@inproceedings{jj-vmed-2025,
    title={JJ-VMed: A Framework for Automated Concepts, Captions and Explainability of Medical Image},
    author={Jaimage Team},
    booktitle={CLEF 2025 Working Notes},
    year={2025},
    series={ImageCLEFmed Caption Lab}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Challenge**: ImageCLEFmed Caption Lab at CLEF 2025
- **Base Model**: LLaVA-LLaMA 3 8B by the LLaVA team
- **Framework**: Built on Transformers, PEFT, and PyTorch

---

**Team**: Jaimage  
**Challenge**: ImageCLEFmed Caption Lab at CLEF 2025  
**Focus**: Medical Image Understanding and Description
