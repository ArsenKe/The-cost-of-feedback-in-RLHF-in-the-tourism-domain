# Cost-Effective RLHF for Tourism Chatbots

This repository contains the implementation and research artifacts for my Master's thesis on developing a cost-effective Reinforcement Learning from Human Feedback (RLHF) system for tourism chatbots.

**Deployed on Hugging Face Space** 

  - https://huggingface.co/spaces/ArsenKe/rlhf-feedback-app
  
**Model pushed to Hugging Face Model**

  Finetuned Model - https://huggingface.co/ArsenKe/MT5_large_finetuned_chatbot
  
  DPO Reward adapters and tokenizers - https://huggingface.co/ArsenKe/MT5_large_dpo_merged
  
## Abstract
This research addresses the challenge of implementing a cost-effective feedback application for customer service chatbots using RLHF. The study explores strategies to build an RLHF chatbot assistant that relies on human feedback while minimizing implementation costs, specifically tailored for the tourism domain. Using Design Science Research (DSR), the project implements and evaluates various cost-reduction techniques including 8-bit quantization, DPO fine-tuning, and LoRA adaptation.

## Key Contributions
- Development of a robust data collection strategy for human feedback
- Implementation of memory-efficient training techniques (8-bit quantization, gradient checkpointing)
- Different model architectures application for dataset  (MT5-Large, Llama .etc)
- Design of an effective reward mechanism using Direct Preference Optimization (DPO)
- End-to-end RLHF pipeline with Firebase integration for real-time feedback

## Methodology

### Main Research Question
**How to develop a cost-effective and high-performing tourism chatbot using RLHF?**

### Technical Approach
- **Model Architecture**: MT5-Large (1.2B parameters) with LoRA adaptation
- **Training Method**: Direct Preference Optimization (DPO)
- **Efficiency Techniques**:
  - 8-bit quantization (BitsAndBytes)
  - Gradient checkpointing
  - Small batch sizes with gradient accumulation
- **Feedback Infrastructure**:
  - Gradio interface for user interactions
  - Firebase Realtime Database for feedback collection
  - DPO training loop


## Model Comparison

| Model               | Parameters | Fine-Tuning Method               | Suitability for Task             |
|---------------------|------------|-----------------------------------|----------------------------------|
| Llama-3.2-1B-Instruct | 1.24B      | LoRA, for causal language modeling | High (instruction-following)     |
| Llama-2-7b-chat-hf  | 6.74B      | LoRA, for chat applications       | Moderate (resource-heavy)        |
| Flan-T5-small       | 80M        | LoRA, for sequence-to-sequence tasks | Low (limited capacity)          |
| Flan-T5-large       | 780M       | Full/LoRA, fully fine-tuned       | High (multilingual)              |
| BERT-base-uncased   | 110M       | Full, for classification/embedding | Low (not generative)            |
| DialoGPT            | 355M       | Full/LoRA, conversational tasks   | Moderate (conversational)        |
| MT5-Large           | 1.2B       | Full, fully fine-tuned            | High (multilingual)              |


### Model Training Details
| Parameter                | Value                     |
|--------------------------|---------------------------|
| Base Model               | MT5-Large                 |
| Fine-tuning Method       | DPO + LoRA                |
| Learning Rate            | 1e-6                      |
| Batch Size               | 1 (per device)            |
| Gradient Accumulation    | 16 steps                  |
| Sequence Length          | 256 tokens                |
| Training Samples         | 40,000                    |
| Evaluation Samples       | 10,000                    |

## Implementation

### Core Components
 **Feedback Collection System**
   - Gradio web interface
   - Firebase Realtime Database integration
   - Quality rating mechanism


### Results

The MT5-Large model achieved consistent training loss reduction:

Training Step	Loss Value
500	10.3892
1000	0.6514
...	...
8000	0.1225

**DPO training showed stable convergence for the 117 feedback comparison dataset with final loss values around 2.7**

**Repository Structure**

├── LLM_finetuning_colab/          # Model training notebooks

├── RLHF_feedback_on_HuggingFaceSpace/  # Feedback interface

├── RLHF_pipeline_automisation/    # Automated training pipeline

├── README.md                      


### Requirements

Python 3.9+

PyTorch 2.0+

Transformers, TRL, PEFT

Firebase Admin SDK

Weights & Biases (for logging)
