# LLM Classification Project

## Project Overview
This project focuses on training language models for sequence classification tasks using advanced techniques like Lora and BitsAndBytes. It's designed to be flexible and easily adaptable to various datasets and models.

## Features
- Utilizes state-of-the-art models like Meta-LLaMA for classification tasks.
- Supports fine-tuning with Lora and BitsAndBytes configurations.
- Easily configurable training parameters through `config.yaml`.

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.x
- PyTorch
- Hugging Face's Transformers library
- Other dependencies listed in `requirements.txt`.

## Configuration
The `config.yaml` file contains various configurations for training:
- Training arguments like batch size, learning rate, etc.
- Lora and BitsAndBytes configurations for advanced model tuning.

## Usage
To train a model, run the `train.py` script with the necessary arguments. You will need to specify the dataset name, the model name, and the maximum sequence length for tokenization. The script allows for flexible configuration and is adaptable to various models and datasets.

Parameters:
- `--dataset_name`: Specify the name of the dataset for training.
- `--model_name`: Indicate the pre-trained model to be used.
- `--max_length`: Define the maximum sequence length for tokenization.

For detailed usage instructions, refer to the comments in the script.


## Scripts
- `train.py`: Main script for training models. It includes functions for model preparation, training, and saving.
- `utils.py`: Contains utility functions for dataset tokenization and computing evaluation metrics.
- `run_scripts.sh`: A Bash script for running training jobs, useful for batch processing or automating multiple training runs.
