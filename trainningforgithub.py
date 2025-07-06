from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_dataset
import torch
import json
import os
import transformers
from transformers import TrainingArguments
import inspect

print("Transformers loaded from:", transformers.__file__)
print("Transformers version:", transformers.__version__)
print("TrainingArguments signature:", inspect.signature(TrainingArguments))
# Set up Hugging Face token for access to gated models
os.environ["HUGGINGFACE_TOKEN"] = ""
# Alternative login method
from huggingface_hub import login
login(token="")

# Use Mistral 7B model
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Load tokenizer with token authentication
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    token=""
)
tokenizer.pad_token = tokenizer.eos_token

# Load model with token authentication
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,
    token=""
)

# Set up LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Shows how many parameters are being trained

# Load datasets from JSONL files
print("Loading datasets...")
train_dataset = load_dataset('json', data_files='C:/Users/Lunga/Desktop/projects/Ai/llama2-chatbot/src/data/simple-illness/train_fixed.jsonl', split='train')
valid_dataset = load_dataset('json', data_files='C:/Users/Lunga/Desktop/projects/Ai/llama2-chatbot/src/data/simple-illness/valid_fixed.jsonl', split='train')
test_dataset = load_dataset('json', data_files='C:/Users/Lunga/Desktop/projects/Ai/llama2-chatbot/src/data/simple-illness/test_fixed.jsonl', split='train')

print(f"Loaded {len(train_dataset)} training examples")
print(f"Loaded {len(valid_dataset)} validation examples")
print(f"Loaded {len(test_dataset)} test examples")

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_valid = valid_dataset.map(tokenize_function, batched=True)

# Remove the original text column and set format to torch
tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_valid = tokenized_valid.remove_columns(["text"])
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask"])
tokenized_valid.set_format("torch", columns=["input_ids", "attention_mask"])

# Use Hugging Face's built-in collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    return_tensors="pt"
)

# Training arguments
training_args = TrainingArguments(
     output_dir="./medical-mistral",
     num_train_epochs=3,
     per_device_train_batch_size=1,
     gradient_accumulation_steps=8,
     learning_rate=1e-4,
     fp16=True,
     save_strategy="epoch",
     logging_steps=10,
     save_total_limit=10,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
)

# Start training
print("Starting training...")
trainer.train()

# Save the fine-tuned model
model.save_pretrained("medical_mistral_final")
tokenizer.save_pretrained("medical_mistral_final")

print("Training complete! Model saved to medical_mistral_final")

# In your scrape_sections function:
def get_block(heading_texts):
    """Extract content blocks based on heading text"""
    for h in soup.find_all(["h1", "h2", "h3", "h4", "strong", "b"]):
        h_text = h.get_text(strip=True).lower()
        if any(txt.lower() in h_text for txt in heading_texts):
            out = []
            
            # Look for treatment content in next siblings
            for sib in h.find_next_siblings():
                if sib.name in ["h1", "h2", "h3", "h4"]:
                    break
                if sib.name in ["p", "li", "ul"]:
                    text = sib.get_text(strip=True)
                    # Filter out boilerplate content
                    if (text and 
                        len(text) > 15 and
                        "official website" not in text.lower() and
                        "back to" not in text.lower()):
                        out.append(text)
            
            content = " ".join(out)
            # Only return if meaningful content was found
            if len(content) > 30:
                return content
    return ""