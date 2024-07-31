"""
Language Model Fine-Tuning Script

This script is designed to fine-tune a pre-trained language model using a custom dataset. 
It provides a complete pipeline for loading data, preparing the dataset, initializing the model 
and tokenizer, setting up training arguments, and conducting the fine-tuning process.

Key features:
1. Loads data from a JSON file
2. Prepares and tokenizes the dataset for training
3. Initializes a pre-trained model and tokenizer (with optional authentication)
4. Configures training arguments
5. Trains the model using the Hugging Face Trainer
6. Saves the fine-tuned model
7. Zips the output directory for easy sharing or deployment

The script uses argparse to allow for flexible command-line configuration of various parameters 
such as the data path, model name, output directory, number of training epochs, batch size, 
learning rate, and more.

It's designed to work with PyTorch and can utilize MPS (Metal Performance Shaders) if available, 
falling back to CPU if not.

Usage:
python script_name.py --data_path path/to/data.json --model_name model_name --output_dir ./output --num_train_epochs 3

For more options, use:
python script_name.py --help
"""

import json
import torch
import argparse
import shutil
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Function to load data from a JSON file
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Function to prepare the dataset for training
def prepare_dataset(data):
    texts = [f"Instruction: {item['instruction']}\nResponse: {item['response']}" for item in data]
    return Dataset.from_dict({"text": texts})

# Function to tokenize the dataset
def tokenize_data(examples, tokenizer):
    model_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Initialize the tokenizer and model
def initialize_model_and_tokenizer(model_name,use_auth_token,auth_token):
    if use_auth_token:
        tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token=auth_token)
        # Add a padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager",use_auth_token=auth_token)
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Add a padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

# Define the training arguments
def define_training_args(output_dir, num_train_epochs, batch_size, save_steps, learning_rate):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        save_steps=save_steps,
        save_total_limit=2,
        learning_rate=learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=200,
        use_mps_device=True,
    )

# Create the Trainer object
def create_trainer(model, args, train_dataset, tokenizer):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

# Function to fine-tune and train the model
def train_model(trainer, output_dir):
    trainer.train()
    trainer.save_model(output_dir)

# Function to zip the output model directory
def zip_output_directory(output_dir):
    shutil.make_archive(output_dir, 'zip', output_dir)

# Main function to orchestrate the training process
def main():
    parser = argparse.ArgumentParser(description="Fine-tune a language model.")
    parser.add_argument('--data_path', type=str, default="data.json", help="Path to the JSON file containing the data.")
    parser.add_argument('--model_name', type=str, default="ybelkada/falcon-7b-sharded-bf16", help="Name of the model to use.")
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save the model.")
    parser.add_argument('--num_train_epochs', type=int, default=1, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=2, help="Training batch size.")
    parser.add_argument('--save_steps', type=int, default=10000, help="Save steps.")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate.")
    parser.add_argument('--use_auth_token', action='store_true', help="Whether to use authentication token.")
    parser.add_argument('--auth_token', type=str, default="", help="Authentication token or credentials.")
    args = parser.parse_args()

    # Set the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(args.model_name,args.use_auth_token,args.auth_token)

    # Move model to the appropriate device
    model.to(device)

    # Load and prepare data
    data = load_data(args.data_path)
    dataset = prepare_dataset(data)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_data(x, tokenizer),
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names
    )

    # Define training arguments
    training_args = define_training_args(args.output_dir, args.num_train_epochs, args.batch_size, args.save_steps, args.learning_rate)

    # Create Trainer
    trainer = create_trainer(model, training_args, tokenized_dataset, tokenizer)

    # Train model
    train_model(trainer, args.output_dir)

    # Zip the output directory
    zip_output_directory(args.output_dir)

if __name__ == "__main__":
    main()