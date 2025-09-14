# Optimized Mistral AI Trainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import os
#helll
def main():
    os.environ["HUGGINGFACE_TOKEN"] = "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"
    from huggingface_hub import login
    login(token="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")
  
    train_file = r'C:\Users\Lunga\Desktop\projects\Ai\llama2-chatbot\src\data\simple-illness\mistra\processed\mts_train_mistral_rtx4070.jsonl'
    valid_file = r'C:\Users\Lunga\Desktop\projects\Ai\llama2-chatbot\src\data\simple-illness\mistra\processed\mts_valid_mistral_rtx4070.jsonl'

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token="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",
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Use 4-bit quantization instead of 8-bit for better LoRA compatibility
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config,
        token="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",
        torch_dtype=torch.float16
    )
    # disable cache to be compatible with gradient checkpointing
    try:
        model.config.use_cache = False
    except Exception:
        pass

    # prepare model for k-bit training (important for 4-bit + LoRA)
    model = prepare_model_for_kbit_training(model)

    # enable gradient checkpointing after preparation
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Ensure only LoRA/adapter params are trainable; freeze the rest
    for name, param in model.named_parameters():
        if ("lora" in name.lower()) or ("adapter" in name.lower()):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Diagnostics: print trainable params and a sample device
    print("Trainable parameters (name, shape):")
    trainable_count = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable_count += p.numel()
            print(f"  {name} {tuple(p.shape)} {p.device}")
    print(f"Total trainable parameters: {trainable_count}")

    print("Loading datasets...")
    train_dataset = load_dataset('json', data_files=train_file, split='train')
    valid_dataset = load_dataset('json', data_files=valid_file, split='train')
    print(f"Loaded {len(train_dataset)} training examples")
    print(f"Loaded {len(valid_dataset)} validation examples")

    def tokenize_function(examples):
        system_prompt = "You are a kind, caring nurse. Answer as if you are helping your patient.\n"
        field = None
        for candidate in ["text", "instruction", "prompt"]:
            if candidate in examples:
                field = candidate
                break
        if not field:
            raise ValueError("No recognized input field in dataset.")
        if isinstance(examples[field], list):
            inputs = [system_prompt + (x if x is not None else "") for x in examples[field]]
        else:
            inputs = [system_prompt + (examples[field] if examples[field] is not None else "")]
        tokenized = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
        # labels for causal LM should be same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True, 
        batch_size=64,  
        num_proc=4      
    )
    tokenized_valid = valid_dataset.map(
        tokenize_function, 
        batched=True, 
        batch_size=64,
        num_proc=4
    )

    for col in ["text", "instruction", "prompt"]:
        if col in tokenized_train.column_names:
            tokenized_train = tokenized_train.remove_columns([col])
        if col in tokenized_valid.column_names:
            tokenized_valid = tokenized_valid.remove_columns([col])

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_valid.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )
 #kill me this is just a trainner 
    training_args = TrainingArguments(
        output_dir="./medical-mistral",
        num_train_epochs=3,
        per_device_train_batch_size=1,  
        gradient_accumulation_steps=8,  
        learning_rate=1e-4,
        fp16=True,
        optim="adamw_torch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=3,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_bf16_supported(),  
        group_by_length=True,              
        report_to=[],
        #report_to=["tensorboard"],
    )
 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Training complete! Saving model...")
    model.save_pretrained("medicalfinal")
    tokenizer.save_pretrained("medicalfinal")
    print("Model saved to medicalfinal")

    model.save_pretrained(r"c:\Users\Lunga\Desktop\projects\Ai\llama2-chatbot\src\data\simple-illness\mistra\Numa")
    tokenizer.save_pretrained(r"c:\Users\Lunga\Desktop\projects\Ai\llama2-chatbot\src\data\simple-illness\mistra\Numa")
    print("✅ Model also saved to Numa folder")

if __name__ == "__main__":
    main()
            if len(content) > 30:
                return content

    return ""
