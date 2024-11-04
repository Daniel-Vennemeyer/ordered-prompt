import json
import time
import torch
import random
import pickle
import argparse
import logging

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

if __name__ == "__main__":
    import yaml
    import easydict
    dataset = 'sst2'

    # corpus_config = yaml.safe_load(open("config/rte.yaml"))
    corpus_config = yaml.safe_load(open(f"/Users/danielvennemeyer/Workspace/NLP/ordered-prompt/config/{dataset}.yaml"))
    cfg = easydict.EasyDict(corpus_config)
    rte = PromptCorpus(**cfg)
    datapoint = rte[0]
    prompts = [datapoint['raw_sequence'] for datapoint in rte]
    roberta.run_prompt_variations(prompts, dataset_name=dataset)


