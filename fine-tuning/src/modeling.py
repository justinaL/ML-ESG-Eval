from datetime import datetime

from transformers import AutoModelForSequenceClassification, TrainingArguments, AutoTokenizer
from datasets import concatenate_datasets, DatasetDict
from src.preprocess import *


class ESGClassifier():
    def __init__(self, params):
        self.tokenizer = AutoTokenizer.from_pretrained(params.base_model, use_fast=True)

    def init_model(self, params):
        print(f'>>> {datetime.now()} Initialise model')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            params.base_model,
            problem_type="multi_label_classification",
            num_labels=len(params.labels),
            id2label=params.id2label,
            label2id=params.label2id,
            attention_probs_dropout_prob=params.attn_dropout, # for mBERT and mE5
            # attention_dropout=params.attn_dropout, # for SBERT
            )

        self.args = TrainingArguments(
            params.out_path,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=params.lr,
            per_device_train_batch_size=params.batch_size,
            per_device_eval_batch_size=params.batch_size,
            num_train_epochs=params.epoch,
            weight_decay=0.01,
            save_total_limit=4,
            load_best_model_at_end=True,
            metric_for_best_model=params.best_model_metric,
            fp16=True
            )

    def preprocess(self, ds, select=True):
        prepared = []
        min_length = min(len(d) for d in ds)

        # load to Dataset Obj and split
        for d in ds:
            d = preprocess(d)
            d = d.train_test_split(0.25)
            prepared.append(d)

        full_train = concatenate_datasets([d['train'] for d in prepared])
        full_val = concatenate_datasets([d['test'] for d in prepared])

        print(f'>>> {datetime.now()} Tokenising')
        ## encode text (concat_content)
        encoded_full_train = tokenise(self.tokenizer, full_train)
        encoded_full_val = tokenise(self.tokenizer, full_val)
        encoded_dataset = DatasetDict({'train':encoded_full_train,'val':encoded_full_val})

        return encoded_dataset, prepared