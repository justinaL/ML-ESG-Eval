import argparse
from datasets import Value, Features
from transformers import pipeline
import pandas as pd
from transformers import Trainer

from src.modeling import ESGClassifier
from src.preprocess import *
from src.train_utils import *
from datetime import datetime

parser = argparse.ArgumentParser(description='ESGClassifier-inference')

# to be updated
parser.add_argument("--trained_model", type=str, default="best_model/bert-base-multilingual-uncased", help="trained model to be tested")
parser.add_argument("--test_file", type=str, default=None, help="test json file")
parser.add_argument("--attn_dropout", type=float, default=0.1, help="model attention dropout value")
parser.add_argument("--out_path", type=str, default="mbert-finetuned-esg-uncased", help="checkpoints save path")

parser.add_argument("--labels", type=str, default='labels.pickle', help="ESG labels")
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--best_model_metric", type=str, default="f1_macro")
parser.add_argument("--epoch", type=int, default=5)
parser.add_argument("--lr", type=int, default=2e-5)


params = parser.parse_args()

params.base_model = params.trained_model

# ======================================================================

lang = params.test_file.split('/')[-1].split('_')[0] # for logging

params.labels = pd.read_pickle(params.labels)
params.id2label = {idx: label for idx, label in enumerate(params.labels)}
params.label2id = {label: idx for idx, label in enumerate(params.labels)}

## define model
model = ESGClassifier(params)

## initialise model
model.init_model(params)

## load training arguments
arguments = torch.load('%s/training_args.bin'%params.trained_model)
arguments.output_dir = params.out_path

trainer = Trainer(
    model.model,
    arguments,
    tokenizer=model.tokenizer,
    compute_metrics=compute_metrics
)

## read json file to pd.DataFrame
test_json = pd.read_json(params.test_file)
if 'google_trans_trunc_concat_content' in test_json.columns:
    test_json['concat_content'] = test_json['google_trans_trunc_concat_content']
elif 'dl_trans_trunc_concat_content' in test_json.columns:
    test_json['concat_content'] = test_json['dl_trans_trunc_concat_content']

## preprocess to Dataset obj and encode text input
test_ds = preprocess(test_json)
test_ds = tokenise(model.tokenizer, test_ds)

## cast labels to float32 (ds not splited, no keys)
feats = test_ds.features.copy()
feats['labels'].feature = Value(dtype='float32')
feats = Features(feats)
test_ds = test_ds.cast(feats)

test_metrics = trainer.predict(test_ds)

print(f'>>> {datetime.now()} Finished Inferencing')

trainer.log_metrics('test_%s'%lang, test_metrics.metrics)
trainer.save_metrics('test_%s'%lang, test_metrics.metrics)