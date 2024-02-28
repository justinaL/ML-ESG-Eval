from datasets import Dataset, concatenate_datasets, DatasetDict, Value, Features
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
import pandas as pd
import numpy as np

def preprocess(ds):
    ## load into Dataset obj
    ds = Dataset.from_pandas(ds)

     ## select columns [input paragraphs, output label(s)]
    ds = ds.select_columns(['concat_content','bin_label'])

    ## rename column and shuffle data
    ds = ds.rename_column('bin_label','labels')
    ds = ds.shuffle(seed=12345)

    return ds

def tokenise(tokenizer, ds):
    return ds.map(lambda examples: tokenizer(examples['concat_content'], padding="max_length", truncation=True, return_tensors='pt'),
                  batched=True
                  )

def cast_to_format(ds, keys, format='float32'):
    for k in keys:
        feats = ds[k].features.copy()
        feats['labels'].feature = Value(dtype=format)
        feats = Features(feats)
        ds[k] = ds[k].cast(feats)

    return ds