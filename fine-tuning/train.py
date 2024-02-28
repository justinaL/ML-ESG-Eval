import argparse, json

import pandas as pd
from datetime import datetime
from transformers import Trainer, set_seed

from src.modeling import ESGClassifier
from src.preprocess import cast_to_format
from src.train_utils import *


parser = argparse.ArgumentParser(description='ESGClassifier-train')

parser.add_argument("--base_model", type=str, default="bert-base-multilingual-uncased", help="Base encoding model")
parser.add_argument("--labels", type=str, default='labels.pickle', help="ESG labels")
parser.add_argument("--select", type=bool, default=False, help="balance dataset or not")
parser.add_argument("--best_model_metric", type=str, default="f1_macro")

parser.add_argument("--out_path", type=str, default="mbert-finetuned-esg-uncased", help="checkpoints save path")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--epoch", type=int, default=5)
parser.add_argument("--lr", type=int, default=2e-5)
parser.add_argument("--attn_dropout", type=float, default=0.1, help="model attention dropout value")
parser.add_argument("--best_model_name", type=str, default="best_models")

params = parser.parse_args()

with open(f'{params.out_path}_training_args.json', 'w') as f:
    json.dump(vars(params), f, indent=2)

# ======================================================================
print(f'>>> {datetime.now()}')

# take too long if load dataset directly from json file
en = pd.read_json('../data/train/en_train.json')
# fr = pd.read_json('../data/train/fr_train.json')
# zh = pd.read_json('../data/train/zh_train.json')

fr = pd.read_json('../data/train/fr_train_google_trans.json')
fr['concat_content'] = fr['google_trans_trunc_concat_content']
zh = pd.read_json('../data/train/zh_train_google_trans.json')
zh['concat_content'] = zh['google_trans_trunc_concat_content']

# params.labels = set([ele for sl in encoded_dataset['train']['ESG_label'] for ele in sl])
params.labels = pd.read_pickle(params.labels)
params.id2label = {idx: label for idx, label in enumerate(params.labels)}
params.label2id = {label: idx for idx, label in enumerate(params.labels)}

out_dir = params.out_path

for seed in [12345,34567,56789]:

    ## set random seed here
    set_seed(seed)
    params.out_path = f'{out_dir}_{seed}_{params.epoch}ep'

    ## define model
    model = ESGClassifier(params)

    ## preprocess datasets
    encoded_dataset,_ = model.preprocess([en,fr,zh], select=params.select)

    ## cast labels to float
    encoded_dataset = cast_to_format(encoded_dataset, ['train','val'], 'float32')
    encoded_dataset.set_format('torch')

    ## start training
    ## initialise model
    model.init_model(params)

    trainer = Trainer(
        model=model.model,
        args=model.args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["val"],
        tokenizer=model.tokenizer,
        compute_metrics=compute_metrics
    )

    print(f'>>> {datetime.now()} Training model with seed {seed}')
    train_metrics = trainer.train()
    trainer.log_metrics('train', train_metrics.metrics)
    trainer.save_metrics('train', train_metrics.metrics)

    today = datetime.today().strftime('%d%m%y')
    trainer.save_model(f'best_models/{params.best_model_name}_{today}_{seed}_{params.epoch}ep')

    print(f'>>> {datetime.now()} Evaluate model')
    eval_metrics = trainer.evaluate()
    trainer.log_metrics('eval', eval_metrics)
    trainer.save_metrics('eval', eval_metrics)

    print(datetime.now())