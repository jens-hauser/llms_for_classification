import argparse
import os
import gc

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from utils import compute_metrics, get_tokenized_dataset

os.environ["WANDB_PROJECT"] = "llm_classification_performances"
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_SILENT "] = "true"
os.environ["WANDB_MODE"] = "offline"

config = yaml.safe_load(open("config.yaml"))

def train_model(dataset_name: str, model_name: str, max_length: int) -> None:
    # Load config and read basic hyperparameters
    model_name_short = model_name.split("/")[-1].replace("-", "_")

    print(f"Prepare training of {model_name_short} on {dataset_name}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get tokenized train and test splits from dataset
    train, eval = get_tokenized_dataset(dataset_name, tokenizer, max_length=max_length)
    num_labels = len(set(train["labels"]))
    if len(eval) > 5000:
        eval = eval.shuffle(seed=42).select(range(5000))

    print(f"Train size: {len(train)}", f"Eval size: {len(eval)}", sep="\n")

    # Load pretrained model
    if model_name.startswith("meta-llama"):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        print(f"Model {model_name} loaded successfully on {model.device}")

        # Set pad token to eos token
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
            lora_dropout=0.1,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        print(f"{model.print_trainable_parameters()}")

    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.bfloat16,
        ).to(device)
        print(f"Model {model_name} loaded successfully on {model.device}")

    # Train model
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"./checkpoints/{model_name_short}_{dataset_name}",
            label_names=["labels"],
            num_train_epochs = 5 if dataset_name != "ag_news" else 3,
            run_name=f"{model_name_short}_{dataset_name}",
            **config["training_arguments"],
        ),
        train_dataset=train,
        eval_dataset=eval,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    trainer.train()

    # Save model
    if model_name.startswith("meta-llama"):
        save_dir = f"./models_peft/{model_name_short}/{dataset_name}/"
        model.save_pretrained(save_dir)
    else:
        save_dir = f"./models/{model_name_short}/{dataset_name}/"
        model.save_pretrained(save_dir)

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-d", type=str, required=True)
    parser.add_argument("--model_name", "-m", type=str, required=True)
    parser.add_argument("--max_length", "-l", type=int, required=False, default=128)
    args = parser.parse_args()

    train_model(args.dataset_name, args.model_name, args.max_length)
