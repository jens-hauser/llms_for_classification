import argparse
import os
import gc

import torch
import yaml
import logging
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from utils import compute_metrics, get_tokenized_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Function to load and prepare model
def prepare_model(model_name, num_labels, device):
    """
    Loads and prepares the model for sequence classification.

    Args:
        model_name (str): Name of the model to be loaded.
        num_labels (int): Number of labels in the classification task.
        device: The device (CPU or GPU) to load the model onto.

    Returns:
        tuple: Returns a tuple of the loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name.startswith("meta-llama"):
        # Special configurations for "meta-llama" models
        bitsandbytes_config = BitsAndBytesConfig(
            load_in_4bit=config['bitsandbytes_config']['load_in_4bit'],
            bnb_4bit_compute_dtype=getattr(torch, config['bitsandbytes_config']['bnb_4bit_compute_dtype']),
            bnb_4bit_use_double_quant=config['bitsandbytes_config']['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_type=config['bitsandbytes_config']['bnb_4bit_quant_type'],
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=getattr(torch, config['bitsandbytes_config']['bnb_4bit_compute_dtype']),
            quantization_config=bitsandbytes_config,
        )
        logger.info(f"Model {model_name} loaded successfully on {model.device}")

        # Set pad token to eos token
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        lora_config = LoraConfig(
            r=config['lora_config']['r'],
            lora_alpha=config['lora_config']['lora_alpha'],
            target_modules=config['lora_config']['target_modules'],
            lora_dropout=config['lora_config']['lora_dropout'],
            bias=config['lora_config']['bias'],
        )

        model = get_peft_model(model, lora_config)
        logger.info(f"{model.print_trainable_parameters()}")
    else:
        # Standard model loading for other model types
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.bfloat16,
        ).to(device)
    
        logger.info(f"Model {model_name} loaded successfully on {model.device}")
    return model, tokenizer

# Function to set up and train the model
def setup_and_train(model, tokenizer, train, eval, dataset_name, model_name_short):
    """
    Sets up training arguments and trains the model.

    Args:
        model: The model to be trained.
        tokenizer: Tokenizer to be used for training.
        train: Training dataset.
        eval: Evaluation dataset.
        dataset_name (str): Name of the dataset.
        model_name_short (str): Shortened model name for directory naming.

    """
    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{model_name_short}_{dataset_name}",
        label_names=["labels"],
        num_train_epochs=5 if dataset_name != "ag_news" else 3,
        run_name=f"{model_name_short}_{dataset_name}",
        **config["training_arguments"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()

    save_dir = f"./models_peft/{model_name_short}/{dataset_name}/" if model_name_short.startswith("meta-llama") else f"./models/{model_name_short}/{dataset_name}/"
    model.save_pretrained(save_dir)
    logger.info(f"Model saved in {save_dir}")

def main(dataset_name, model_name, max_length):
    """
    Main function to handle the training pipeline.

    Args:
        dataset_name (str): Name of the dataset to train on.
        model_name (str): Name of the model to be used.
        max_length (int): Maximum length for tokenization.
    """
    try:
        logger.info(f"Preparing to train model {model_name} on dataset {dataset_name}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train, eval = get_tokenized_dataset(dataset_name, tokenizer, max_length=max_length)
        
        num_labels = len(set(train["labels"]))
        model, tokenizer = prepare_model(model_name, num_labels, device)

        setup_and_train(model, tokenizer, train, eval, dataset_name, model_name.split("/")[-1].replace("-", "_"))
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-d", type=str, required=True)
    parser.add_argument("--model_name", "-m", type=str, required=True)
    parser.add_argument("--max_length", "-l", type=int, default=128)
    args = parser.parse_args()

    main(args.dataset_name, args.model_name, args.max_length)