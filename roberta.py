import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset, DatasetDict, Dataset
import os
import json
from scipy.stats import pearsonr, spearmanr


class Roberta():
    def __init__(self, dataset_name) -> None:
        # Define file paths for saving/loading
        self.dataset_name = dataset_name
        self.model_dir = f"./saved_model_{dataset_name}"

        # Step 1: Load and Preprocess the Specified Dataset
        dataset = self.load_custom_dataset(dataset_name)
        
        # Determine the number of labels based on the dataset
        if dataset_name in ["sst2", "cr", "mr", "mpqa"]:
            num_labels = 2  # Binary classification
        elif dataset_name in ["sst5", "subj", "agnews", "dbpedia"]:
            num_labels = len(set(dataset['train']['label']))
        elif dataset_name in ["cb", "rte"]:
            num_labels = 3 if dataset_name == "CB" else 2

        # Step 3: Load and Train RoBERTa (or load saved model weights if they exist)
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

        self.model = self.initialize_model(num_labels)

        # Load tokenizer and pre-process the data for RoBERTa
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        def tokenize_function(examples):
            return self.tokenizer(examples["sentence"] if "sentence" in examples else examples["text"], 
                            padding="max_length", truncation=True, max_length=128)

        train_data = dataset['train'].map(tokenize_function, batched=True)
        self.test_data = dataset['validation' if 'validation' in dataset else 'test'].map(tokenize_function, batched=True)

        # Convert labels
        train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        self.test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=self.test_data,
            compute_metrics=self.compute_metrics,
        )

        # Only train if there isn't a saved model already
        if not os.path.exists(self.model_dir):
            print("Training model...")
            self.trainer.train()
            self.trainer.save_model(self.model_dir)  # Save the model after training




    def initialize_model(self, num_labels):
            if os.path.exists(self.model_dir):
                print("Loading model from saved checkpoint...")
                model = RobertaForSequenceClassification.from_pretrained(self.model_dir)
            else:
                model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)
            return model

    def kl(self, distp, distq):
        total_sum = 0.0
        for p, q in zip(distp, distq):
            total_sum += (-1.0 * p * np.log(q / p))
        return total_sum

    def cal_entropy(self, dist):
        return sum([-p * np.log(p + 1e-10) for p in dist])

    # Updated Evaluate Prompt Variations Function with Accuracy Calculation
    def evaluate_prompt_variations(self, prompt_variations, prompt_labels, topk=24):
        pipeline_model = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        entropys = []
        acc = []  # Stores accuracy for each prompt

        for i, prompt in enumerate(prompt_variations):
            predictions = pipeline_model(prompt)
            
            # Extract predicted labels and calculate entropy
            probs = np.array([pred["score"] for pred in predictions])
            predicted_label = np.argmax(probs)
            true_label = int(prompt_labels[i])
            
            # Check accuracy for this prompt
            is_correct = int(predicted_label == true_label)
            acc.append(is_correct)  # Append 1 for correct, 0 for incorrect

            # Calculate entropy for this prompt
            entropy = self.cal_entropy(probs)
            entropys.append(entropy)

        result = [entropys, acc]
        return result


    # Helper function to load dataset based on the name
    def load_custom_dataset(self, dataset_name):
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

    def compute_metrics(self, eval_pred):
        try:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = accuracy_score(labels, predictions)
            # precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions)
            return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}
        except Exception as e:
            pass

    def run_prompt_variations(self, prompt_info):
        # Step 4: Evaluate RoBERTa Performance on Test Data
        # roberta_results = self.trainer.evaluate()

        prompt_variations = []
        prompt_labels = []
        for prompt in prompt_info:
            prompt_variation, prompt_label = prompt
            prompt_variations.append(prompt_variation)
            prompt_labels.append(prompt_label)

        # Step 5: Compare Probing Sequence Performance with RoBERTa
        variation_results = self.evaluate_prompt_variations(prompt_variations, prompt_labels)

        # print("RoBERTa Results:", roberta_results)
        print("Top Probing Prompts:", variation_results)

        return variation_results
