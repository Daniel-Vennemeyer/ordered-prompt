import json
import time
import torch
import random
import pickle
import argparse
import logging
import numpy as np

from collections import defaultdict
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from itertools import permutations
from utils import corpus_sampling, create_prompt
import roberta

logger = logging.getLogger()
logger.setLevel(logging.ERROR)
import os

class PromptCorpus:
    def __init__(self, train_data_path="data/train.jsonl",
                 test_data_path="data/dev.jsonl", tokenizer_path='distilgpt2',
                 n_shot=10, label_mapping={0: "negative", 1: "positive"},
                 corpus_params={"sentence_1_str": "", "sentence_2_str": "", "label_str": ""},
                 template="f'Review: {sentence_1}\nSentiment: {label_text}\n'",
                 sample_mode="balance", permutation_max_size=24, sentence_pair=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.kshot = n_shot
        self.max_sequence_length = 1022

        self.label_mapping = label_mapping

        # Initialize restricted token to prevent unwanted tokens
        self.restricted_token = []
        for label_str in self.label_mapping.values():
            label_index = self.tokenizer.encode(f" {label_str}")
            assert len(label_index) == 1, "Label should be a single token."
            self.restricted_token += label_index
        self.restricted_token = tuple(self.restricted_token)

        full_train_data = self.load_jsonl(train_data_path)
        self.train_data = corpus_sampling(full_train_data, kshot=self.kshot, mode=sample_mode,
                                          label_str=corpus_params["label_str"])
        self.test_data = self.load_jsonl(test_data_path)
        self.template = template
        self.sentence_pair = sentence_pair
        self.corpus_params = corpus_params
        self.permutation_max_size = permutation_max_size

        logger.info(f"{self.kshot}-shot, label_mapping: {label_mapping}, template: {template}")
        self._cache = {}

    def __len__(self):
        return len(self.test_data)

    @staticmethod
    def load_jsonl(fp):
        data = []
        with open(fp) as fin:
            for i, line in enumerate(fin):
                decoded = json.loads(line)
                decoded["index"] = i
                data.append(decoded)
        return data

    # Generate permutations for training prompts based on the specified max count
    @staticmethod
    def permute_train_prompts(train_prompts, max_count=24):
        if len(train_prompts) > 4:
            subset = set()
            while len(subset) < max_count:
                subset.add(tuple(random.sample(train_prompts, len(train_prompts))))
        else:
            train_prompts_permutation = list(permutations(train_prompts))
            subset = random.sample(train_prompts_permutation, min(len(train_prompts_permutation), max_count))
        return [''.join(elem) for elem in subset]

    # Add tokens incrementally for test example generation
    @staticmethod
    def test_data_plus_one_token(corpus_data_point, text_str="sentence", label_str="label"):
        text = corpus_data_point[text_str].split()
        label = corpus_data_point[label_str]
        augmented_test_data = [{"sentence": ' '.join(text[:i]), "label": label, "index": corpus_data_point["index"]} for i in range(1, len(text)+1)]
        return augmented_test_data

    def __getitem__(self, item):
        train_prompts = []
        label_str = self.corpus_params["label_str"]
        sentence_1_str = self.corpus_params["sentence_1_str"]
        if self.sentence_pair:
            sentence_2_str = self.corpus_params["sentence_2_str"]

        train_labels = []
        for data in self.train_data:
            train_sentence = (data[sentence_1_str], data.get(sentence_2_str)) if self.sentence_pair else (data[sentence_1_str],)
            train_label = data[label_str]
            train_labels.append(train_label)
            train_label_text = self.label_mapping[train_label]
            prompt = create_prompt(template=self.template, sentence=train_sentence, label_text=train_label_text, test=False, sentence_pair=self.sentence_pair)
            train_prompts.append(prompt)

        if "train_prompts_permutation" not in self._cache:
            train_prompts_permutation = self.permute_train_prompts(train_prompts, max_count=self.permutation_max_size)
            self._cache["train_prompts_permutation"] = train_prompts_permutation
        else:
            train_prompts_permutation = self._cache["train_prompts_permutation"]

        test_sentence = (self.test_data[item][sentence_1_str], self.test_data[item].get(sentence_2_str)) if self.sentence_pair else (self.test_data[item][sentence_1_str],)
        test_label = self.test_data[item][label_str]
        test_label_text = self.label_mapping[test_label]
        test_sequence = create_prompt(template=self.template, sentence=test_sentence, label_text=test_label_text, test=True, sentence_pair=self.sentence_pair)

        input_sequences, input_sequences_prompt, raw_sequences = [], [], []

        for train_sequence in train_prompts_permutation:
            combined_sequence = ''.join([train_sequence, test_sequence]).strip()
            input_sequence = self.tokenizer.encode(combined_sequence, add_special_tokens=True)[-self.max_sequence_length:]
            input_sequences.append(torch.LongTensor(input_sequence))

            input_sequence_prompt = self.tokenizer.encode(train_sequence, add_special_tokens=True)[-self.max_sequence_length:]
            input_sequences_prompt.append(torch.LongTensor(input_sequence_prompt))

            raw_sequences.append(combined_sequence)

        return {
            "input_sequence": torch.stack(input_sequences, dim=0),
            "label": test_label,
            "raw_sequence": raw_sequences,
            "train_metadata": self.train_data,
            "test_index": self.test_data[item]["index"],
            "input_sequences_prompt": torch.stack(input_sequences_prompt, dim=0)
        }
    
# Updated Evaluate Prompt Variations Function with Accuracy Calculation
from scipy.stats import pearsonr, spearmanr
def aggregate_results(final_result, num_subsets=256, topk=4):
    acc = [i/num_subsets for i in final_result[1]][:24]
    entropys = [i for i in final_result[0]][:24]

    # Convert accuracy list to percentages
    acc_mean = np.mean(acc)
    acc_std = np.std(acc)

    # Calculate correlations between entropys and acc
    pearsonr_metric, spearmanr_metric = pearsonr(entropys, acc), spearmanr(entropys, acc)
    print("Pearson Correlation:", pearsonr_metric, "Spearman Correlation:", spearmanr_metric)

    # Sort by entropy and select top-k prompts
    gg = list(zip(entropys, acc))
    gg.sort(key=lambda x: x[0], reverse=True)
    if len(gg) == 2:
        print(f"1-shot case, only two examples, change topk from {topk} to 1")
        topk = 1
    assert len(gg) > topk, f"Total permutations are less than {topk}"
    subset_acc = [elem[1] for elem in gg[:topk]]
    
    # Statistics for top-k subset
    subset_acc_mean = np.mean(subset_acc)
    subset_acc_std = np.std(subset_acc)
    print(f"Before: mean accuracy {acc_mean}, std {acc_std}")
    print(f"After: mean accuracy {subset_acc_mean}, std {subset_acc_std}")

    result = {
        "acc_stats": (acc_mean, acc_std),
        "topk_acc_stats": (subset_acc_mean, subset_acc_std),
        "topk": topk,
        "entropys": entropys,
        "acc": acc,
        "pearsonr_corr": pearsonr_metric,
        "spearmanr_corr": spearmanr_metric,
    }
    return result

def append_results(final_result, result_i):
    # result_i = sorted(result_i, key=lambda x: x[0])
    sorted_result = sorted(zip(result_i[0], result_i[1]), key=lambda x: x[0])
    entropy, acc = zip(*sorted_result)
    entropy, acc = list(entropy), list(acc)   
    for i, prompt in enumerate(result_i[0]):
        if final_result[0]:
            final_result[0][i] += entropy[i] # Stores entropy for each prompt
            final_result[1][i] += acc[i] # Stores accuracy for each prompt
        else:
            final_result = [entropy, acc]
            break
    return final_result


if __name__ == "__main__":
    import yaml
    import easydict
    dataset = 'sst2'

    corpus_config = yaml.safe_load(open(f"config/{dataset}.yaml"))
    # corpus_config = yaml.safe_load(open(f"/Users/danielvennemeyer/Workspace/NLP/ordered-prompt/config/{dataset}.yaml"))
    cfg = easydict.EasyDict(corpus_config)
    final_results = [[],[]]
    num_subsets = 256

    roberta_model = roberta.Roberta(dataset)
    for i in range(num_subsets):
        rte = PromptCorpus(**cfg)
        datapoint = rte[0]
        prompts = [(datapoint['raw_sequence'], datapoint['label']) for datapoint in rte]
        result_i = roberta_model.run_prompt_variations(prompts)
        final_results = append_results(final_results, result_i)
        print(f"Permutation set {i}: {result_i[0][:24]}")

    final_results = aggregate_results(final_results, num_subsets=num_subsets)
    with open(f"evaluation_results/{dataset}_results.json", "w") as entropy_log:
        json.dump(final_results, entropy_log)


