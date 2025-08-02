
# MindSlate: Fine-tuned Gemma-3B for Personal Knowledge Management

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="250"/>](https://github.com/unslothai/unsloth)

## Model Description

**MindSlate** is a fine-tuned version of Google's Gemma-3B model, optimized for personal knowledge management tasks including flashcard generation, reminder processing, content summarization, and task management. The model was trained using Unsloth's efficient fine-tuning techniques for 2x faster training.

- **Architecture**: Gemma-3B with LoRA adapters
- **Model type**: Causal Language Model
- **Fine-tuning method**: 4-bit QLoRA
- **Languages**: English
- **License**: Apache 2.0
- **Developed by**: [Srinivas Nampalli](https://www.linkedin.com/in/srinivas-nampalli/)
- **Finetuned from**: [unsloth/gemma-3b-E2B-it-unsloth-bnb-4bit](https://huggingface.co/unsloth/gemma-3b-E2B-it-unsloth-bnb-4bit)

## Model Sources

- **Repository**: [https://github.com/Srinivasmec26/MindSlate](https://github.com/Srinivasmec26/MindSlate)
- **Base Model**: [unsloth/gemma-3b-E2B-it-unsloth-bnb-4bit](https://huggingface.co/unsloth/gemma-3b-E2B-it-unsloth-bnb-4bit)

## Uses

### Direct Use
MindSlate is designed for:
- Automatic flashcard generation from study materials
- Intelligent reminder creation
- Content summarization
- Task extraction and organization
- Personal knowledge base management

### Downstream Use
Can be integrated into:
- Educational platforms
- Productivity apps
- Note-taking applications
- Personal AI assistants

### Out-of-Scope Use
Not suitable for:
- Medical or legal advice
- High-stakes decision making
- Generating factual content without verification

## How to Get Started

```python
from unsloth import FastLanguageModel
import torch

# Load model with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Srinivasmec26/MindSlate",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Set chat template
tokenizer = FastLanguageModel.get_chat_template(
    tokenizer,
    chat_template="gemma",  # Use "chatml" or other templates if needed
)

# Create prompt
messages = [
    {"role": "user", "content": "Convert to flashcard: Neural networks are computational models..."},
]

# Generate response
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    **inputs, 
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
)
print(tokenizer.decode(outputs[0]))
```

## Training Details

### Training Data
The model was fine-tuned on a combination of structured datasets:

1. **Flashcards Dataset** (400 items):
```bibtex
@misc{educational_flashcards_2025,
  title = {Multicultural Educational Flashcards Dataset},
  author = {Srinivas, Yathi Pachauri,  Swarnim Gupta},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/Srinivasmec26/Educational-Flashcards-for-Global-Learners}
}

```

2. **Reminders Dataset** (100 items):
- *Private collection of contextual reminders*
- Format: {"input": "Meeting with team", "output": {"time": "2025-08-15 14:00", "location": "Zoom"}}
```bibtex
@misc{educational_flashcards_2025,
  title = {Multicultural Educational Flashcards Dataset},
  author = {Srinivas, Yathi Pachauri,  Swarnim Gupta},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/Srinivasmec26/Educational-Flashcards-for-Global-Learners}
}

```

3. **Summaries Dataset** (100 items):
- *Academic paper abstracts and summaries*
- Collected from arXiv and academic publications
```bibtex
@misc{knowledge_summaries_2025,
  title = {Multidisciplinary-Educational-Summaries},
  author = {Srinivas Nampalli, Yathi Pachauri, Swarnim Gupta},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/Srinivasmec26/Multidisciplinary-Educational-Summaries}
}
```

4. **Todos Dataset** (100 items):
```bibtex
@misc{academic_todos_2025,
   title = {Structured To-Do Lists for Learning and Projects},
  author = {Nampalli Srinivas, Yathi Pachauri, Swarnim Gupta},
  year = {2025},
  publisher = {Hugging Face},
  version   = {1.0},
  url = {https://huggingface.co/datasets/Srinivasmec26/Structured-Todo-Lists-for-Learning-and-Projects}
}

```

### Training Procedure
- **Preprocessing**: Standardized into `### Input: ... \n### Output: ...` format
- **Framework**: Unsloth 2025.8.1 + Hugging Face TRL
- **Hardware**: Tesla T4 GPU (16GB VRAM)
- **Training Time**: 51 minutes for 3 epochs
- **LoRA Configuration**:
  ```python
  r=64,           # LoRA rank
  lora_alpha=128, # LoRA scaling factor
  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"],
  ```
- **Optimizer**: AdamW 8-bit
- **Learning Rate**: 2e-4 with linear decay

## Evaluation
*Comprehensive benchmark results will be uploaded in v1.1. Preliminary metrics:*

| Metric               | Value  |
|----------------------|--------|
| **Training Loss**    | 0.1284 |
| **Perplexity**       | TBD    |
| **Task Accuracy**    | TBD    |
| **Inference Speed**  | 42 tokens/sec (T4) |

## Technical Specifications

| Parameter            | Value               |
|----------------------|---------------------|
| Model Size           | 3B parameters       |
| Quantization         | 4-bit (bnb)         |
| Max Sequence Length  | 2048 tokens         |
| Fine-tuned Params    | 1.66% (91.6M)       |
| Precision            | BF16/FP16 mixed     |
| Architecture         | Transformer Decoder |

## Citation

```bibtex
@misc{mindslate2025,
  author = {Srinivas Nampalli },
  title = {MindSlate: Efficient Personal Knowledge Management with Gemma-3B},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Srinivasmec26/MindSlate}},
  note = {Fine-tuned using Unsloth for efficient training}
}
```

## Acknowledgements
- [Unsloth](https://github.com/unslothai/unsloth) for 2x faster fine-tuning
- Google for the [Gemma 3n](https://huggingface.co/sparkreaderapp/gemma-3n-E2B-it) base model
- Hugging Face for [TRL](https://huggingface.co/docs/trl) library

## Model Card Contact
For questions and collaborations:
- Srinivas Nampalli: [LinkedIn](https://www.linkedin.com/in/srinivas-nampalli/)
- Srinivas26k: [Github](https://github.com/Srinivas26k/MindSlate)
