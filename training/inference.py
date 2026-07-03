import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FINETUNED_MODEL = "../outputs/merged_model" # Assuming merged model is here
TEST_PROMPTS = [
    "Mera breakup ho gaya hai, bahut rona aa raha hai. Kundli dekh kar batao kab sab theek hoga?",
    "I have been feeling very sick for 3 days. What does my kundli say about this illness?",
    "Mera Canada visa kab aayega? Mujhe exact date batao.",
    "Kya main stock market mein 10 lakh invest karun? Kundli mein profit hai ya loss?"
]

def generate_response(model, tokenizer, prompt, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response

def main():
    system_prompt = "You are Vedaz's AI Vedic astrologer. You give compassionate, balanced, non-fatalistic guidance."
    
    print("Loading Base Model...")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        base_model.eval()
    except Exception as e:
        print(f"Error loading base model (might be OOM or network): {e}")
        base_model = None
        
    print("Loading Fine-tuned Model...")
    try:
        if os.path.exists(FINETUNED_MODEL):
            ft_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
            ft_model = AutoModelForCausalLM.from_pretrained(
                FINETUNED_MODEL,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            ft_model.eval()
        else:
            print("Fine-tuned model not found. Run training first.")
            ft_model = None
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        ft_model = None

    os.makedirs("../outputs/samples", exist_ok=True)
    
    with open("../outputs/samples/comparison.txt", "w", encoding="utf-8") as f:
        for i, prompt in enumerate(TEST_PROMPTS):
            f.write(f"--- Prompt {i+1} ---\n")
            f.write(f"User: {prompt}\n\n")
            
            if base_model:
                base_resp = generate_response(base_model, base_tokenizer, prompt, system_prompt)
                f.write(f"Base Model:\n{base_resp}\n\n")
            
            if ft_model:
                ft_resp = generate_response(ft_model, ft_tokenizer, prompt, system_prompt)
                f.write(f"Fine-Tuned Model:\n{ft_resp}\n\n")
                
            f.write("="*50 + "\n\n")
            print(f"Processed prompt {i+1}")

    print("Inference complete. Check outputs/samples/comparison.txt")

if __name__ == "__main__":
    main()
