import collections, argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.modeling import SKClassifier
from src.utils import *

parser = argparse.ArgumentParser(description='ESGClassifier-off-the-shelf')

parser.add_argument("--pickle_path", type=str, default="pickles/", help="Path to pickle files")
parser.add_argument("--out_path", type=str, default="off-the-shelf-output", help="Checkpoints save path")
parser.add_argument("--train_type", type=str, default="all", help="Train model with all data or not")

params = parser.parse_args()

with open(f'{params.out_path}/training_args.json', 'w') as f:
    json.dump(vars(params), f, indent=2)

# ======================================================================

# read pickle files (incl. embeddings)
en_train_df = pd.read_pickle(f'{params.pickle_path}/en_train.pkl')
fr_train_df = pd.read_pickle(f'{params.pickle_path}/fr_train.pkl')
zh_train_df = pd.read_pickle(f'{params.pickle_path}/zh_train.pkl')

en_embeddings, en_labels = reshape_input(en_train_df['embeddings']), reshape_input(en_train_df['bin_label'])
fr_embeddings, fr_labels = reshape_input(fr_train_df['embeddings']), reshape_input(fr_train_df['bin_label'])
zh_embeddings, zh_labels = reshape_input(zh_train_df['embeddings']), reshape_input(zh_train_df['bin_label'])

if params.train_type == "all":
    train_x = np.concatenate([en_embeddings, fr_embeddings, zh_embeddings])
    train_y = np.concatenate([en_labels, fr_labels, zh_labels])
else:
    train_x = np.concatenate([en_embeddings, fr_embeddings])
    train_y = np.concatenate([en_labels, fr_labels])


model = SKClassifier(train_x=train_x, train_y=train_y)

for seed in [12345,23456,34567]:
    print(f'>>> TRAINING SEED: {seed}')
    svm_classifier = model.train(model_name='SVM', seed=seed)

    save_pickle(svm_classifier, f'{params.out_path}/svm_classifier_{seed}.pickle')