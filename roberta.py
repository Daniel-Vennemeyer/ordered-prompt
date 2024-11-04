import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset, DatasetDict, Dataset
import os
import json


# Step 2: Define a Probing Sequence Function
def compute_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10))

def evaluate_prompt_variations(prompt_variations, model, tokenizer, test_data, dataset_name):
    entropies = []
    for prompt in prompt_variations:
        pipeline_model = pipeline("text-classification", model=model, tokenizer=tokenizer)
        predictions = pipeline_model(prompt)
        
        probs = np.array([pred["score"] for pred in predictions])
        entropy = compute_entropy(probs)
        entropies.append((prompt, entropy))
    
    # Save the evaluation results
    with open(f"evaluation_results/{dataset_name}_results.json", "w") as entropy_log:
        json.dump(sorted(entropies, key=lambda x: x[1]), entropy_log)
    # Select the top-k prompts by lowest entropy
    top_k_prompts = sorted(entropies, key=lambda x: x[1])[:4]

    return top_k_prompts

# Helper function to load dataset based on the name
def load_custom_dataset(dataset_name):
    if dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2")
    elif dataset_name == "agnews":
        dataset = load_dataset("ag_news")
    elif dataset_name == "dbpedia":
        dataset = load_dataset("dbpedia_14")
    elif dataset_name == "cb":
        dataset = load_dataset("super_glue", "cb")
    elif dataset_name == "rte":
        dataset = load_dataset("super_glue", "rte")
    else:
        raise ValueError(f"Dataset {dataset_name} not currently supported.")
    return dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

def run_prompt_variations(prompt_variations, dataset_name="sst2"):
    # Define file paths for saving/loading
    model_dir = f"./saved_model_{dataset_name}"

    # Step 3: Load and Train RoBERTa (or load saved model weights if they exist)
    def initialize_model(num_labels):
        if os.path.exists(model_dir):
            print("Loading model from saved checkpoint...")
            model = RobertaForSequenceClassification.from_pretrained(model_dir)
        else:
            model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)
        return model

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        logging_dir='./logs',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    # Step 1: Load and Preprocess the Specified Dataset
    dataset = load_custom_dataset(dataset_name)
    
    # Determine the number of labels based on the dataset
    if dataset_name in ["sst2", "cr", "mr", "mpqa"]:
        num_labels = 2  # Binary classification
    elif dataset_name in ["sst5", "subj", "agnews", "dbpedia"]:
        num_labels = len(set(dataset['train']['label']))
    elif dataset_name in ["cb", "rte"]:
        num_labels = 3 if dataset_name == "CB" else 2
    
    model = initialize_model(num_labels)

    # Load tokenizer and pre-process the data for RoBERTa
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def tokenize_function(examples):
        return tokenizer(examples["sentence"] if "sentence" in examples else examples["text"], 
                         padding="max_length", truncation=True, max_length=128)

    train_data = dataset['train'].map(tokenize_function, batched=True)
    test_data = dataset['validation' if 'validation' in dataset else 'test'].map(tokenize_function, batched=True)

    # Convert labels
    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
    )

    # Only train if there isn't a saved model already
    if not os.path.exists(model_dir):
        print("Training model...")
        trainer.train()
        trainer.save_model(model_dir)  # Save the model after training

    # Step 4: Evaluate RoBERTa Performance on Test Data
    roberta_results = trainer.evaluate()

    # Step 5: Compare Probing Sequence Performance with RoBERTa
    selected_prompts = evaluate_prompt_variations(prompt_variations, model, tokenizer, test_data, dataset_name)

    print("RoBERTa Results:", roberta_results)
    print("Top Probing Prompts:", selected_prompts)
