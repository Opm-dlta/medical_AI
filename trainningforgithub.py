import os
os.environ["CUDA_HOME"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1"
os.environ["PATH"] = os.environ["PATH"] + ";C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin"

import sys
if sys.version_info >= (3, 13):
    # Workaround for Python 3.13 __class__ issue
    import os
    os.environ["PYTORCH_DISABLE_CUDA_MEMORY_CACHING"] = "1"

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
import sys
import platform

# Python 3.13 workaround for torch.nn.Module.train __class__ NameError
if sys.version_info >= (3, 13):
    import torch.nn as _nn

    _orig_train = _nn.Module.train

    def _patched_train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode should be boolean")
        object.__setattr__(self, "training", mode)
        for module in self.children():
            module.train(mode)
        return self

    _nn.Module.train = _patched_train  # type: ignore[attr-defined]
#helll
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

def _print_env_diag():
    try:
        print("==== Environment diagnostics ====")
        print(f"Python: {sys.version.split()[0]} ({platform.system()} {platform.release()})")
        print(f"Torch: {getattr(torch, '__version__', None)} | CUDA: {getattr(torch.version, 'cuda', None)}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                print(f"GPU: {torch.cuda.get_device_name(0)} | count: {torch.cuda.device_count()}")
                print(f"cuDNN: {torch.backends.cudnn.version()} | enabled: {torch.backends.cudnn.enabled}")
            except Exception:
                pass
    except Exception:
        pass
def main():
   
    hf_token = (
        os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )
    if hf_token:
        print("Using Hugging Face token from environment.")
    else:
        print("No Hugging Face token found in environment; attempting anonymous access (may fail for gated/private models)")
  
    train_file = r'C:\Users\Lunga\Desktop\projects\Ai\llama2-chatbot\src\data\simple-illness\mistra\processed\mts_train_mistral_rtx4070.jsonl'
    valid_file = r'C:\Users\Lunga\Desktop\projects\Ai\llama2-chatbot\src\data\simple-illness\mistra\processed\mts_valid_mistral_rtx4070.jsonl'

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    token_kwargs = {"token": hf_token} if hf_token else {}

    # Allow training on CPU or GPU; no early exit

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        **token_kwargs,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Check bitsandbytes availability for 4-bit quantization (QLoRA)
    have_bnb = True
    try:
        import bitsandbytes as _bnb  # noqa: F401
    except Exception as _e:
        have_bnb = False
        print("\nWARNING: bitsandbytes not importable. 4-bit quantization will be disabled.")
        print("Reason:", str(_e))
        if os.getenv("ALLOW_NO_BNB", "0").lower() not in ("1", "true", "yes", "y"):
            print("Install 'bitsandbytes-windows' (native Windows) or set ALLOW_NO_BNB=1 to proceed without 4-bit.")
            return

    # Use 4-bit quantization instead of 8-bit for better LoRA compatibility (if available)
    quant_config = None
    if have_bnb:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # If running on CPU, disable 4-bit quantization to avoid bitsandbytes/CUDA requirements
    if not torch.cuda.is_available():
        quant_config = None
    # Choose device map: use cuda:0 if available, else CPU (quiet)
    use_cuda = torch.cuda.is_available()
    device_map = {"": "cuda:0"} if use_cuda else "cpu"

    # Speed flags on NVIDIA GPUs (safe defaults)
    if use_cuda:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **token_kwargs,
    )
    try:
        model.config.use_cache = False
        # Disable problematic features for Python 3.13
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    except Exception as e:
        print(f"Warning: Could not configure model: {e}")

    # Prepare model for training
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception as e:
        print(f"Warning: Could not prepare model for kbit training: {e}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    if use_cuda and os.getenv("TORCH_COMPILE", "0") in ("1", "true", "True"):
        mode = os.getenv("TORCH_COMPILE_MODE", "max-autotune")
        try:
            model = torch.compile(model, mode=mode, fullgraph=False)
            print(f"torch.compile enabled (mode={mode})")
        except Exception as e:
            print("torch.compile not applied:", str(e))

    devices = set()
    for name, p in model.named_parameters():
        if p.requires_grad:
            devices.add(str(p.device))
    print("Trainable param devices:", devices)
    if hasattr(model, "hf_device_map"):
        try:
            print("hf_device_map devices:", set(model.hf_device_map.values()))
        except Exception:
            pass

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
    # Prefer bf16 on supported GPUs; else use fp16 on CUDA, and float32 on CPU
    try:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        use_bf16 = False
    fp16_flag = (torch.cuda.is_available() and not use_bf16)

    # Use fused AdamW on GPU for throughput
    optim_name = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

    training_args = TrainingArguments(
        output_dir="./medical-mistral",
        num_train_epochs=3,
        per_device_train_batch_size=1,  
        gradient_accumulation_steps=8,  # Reverted back to original
        learning_rate=1e-4,
        fp16=fp16_flag,
        bf16=use_bf16,
        optim=optim_name,
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=3,  # Reverted back to original
        dataloader_pin_memory=True,  # Reverted back to original
        dataloader_num_workers=2,  # Reverted back to original
        gradient_checkpointing=True,
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
