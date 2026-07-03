import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

# Configuration
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 2048
DTYPE = None # None for auto detection
LOAD_IN_4BIT = True
OUTPUT_DIR = "outputs/checkpoints"
MERGED_DIR = "outputs/merged_model"
TRAIN_DATA = "data/splits/train.jsonl"
VAL_DATA = "data/splits/val.jsonl"

def format_chat_template(examples):
    # Unsloth supports the standard HF conversational format, we just need to ensure the structure is correct
    # The ShareGPT or standard HF tokenizer apply_chat_template does this.
    pass # If using FastLanguageModel.get_chat_template, this might not be needed as dataset mapping handles it.

def main():
    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )
    
    # Setup Chat Template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5", # Standard template for Qwen
    )
    
    # Define mapping function for formatting dataset
    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
            texts.append(text)
        return { "text" : texts, }
        
    print("Loading datasets...")
    train_dataset = load_dataset("json", data_files=TRAIN_DATA, split="train")
    val_dataset = load_dataset("json", data_files=VAL_DATA, split="train")
    
    print("Formatting datasets...")
    train_dataset = train_dataset.map(formatting_prompts_func, batched = True)
    val_dataset = val_dataset.map(formatting_prompts_func, batched = True)
    
    print("Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    
    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 3, # Use 1 for quick testing, 3 for full fine-tuning
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = OUTPUT_DIR,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=10,
            load_best_model_at_end=True,
        ),
    )
    
    print("Starting Training...")
    trainer_stats = trainer.train()
    
    print("Training completed. Saving checkpoint...")
    model.save_pretrained(f"{OUTPUT_DIR}/final_lora")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_lora")
    
    print("Merging adapters with base model...")
    # This creates a full merged model that is easier for inference engines like vLLM
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method = "merged_16bit",)
    print(f"Model successfully saved and merged to {MERGED_DIR}")

if __name__ == "__main__":
    main()
