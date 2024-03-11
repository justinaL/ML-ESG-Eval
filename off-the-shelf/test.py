import collections, argparse
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.utils import *

parser = argparse.ArgumentParser(description='ESGClassifier-off-the-shelf')

parser.add_argument("--pickle_path", type=str, default="pickles/", help="Path to pickle files")
parser.add_argument("--classifier_path", type=str, default="off-the-shelf-output", help="Trained classifier path")

params = parser.parse_args()

with open(f'{params.out_path}/testing_args.json', 'w') as f:
    json.dump(vars(params), f, indent=2)

# ======================================================================

en_test_df = pd.read_pickle(f'{params.pickle_path}/en_test.pkl')
fr_test_df = pd.read_pickle(f'{params.pickle_path}/fr_test.pkl')
zh_test_df = pd.read_pickle(f'{params.pickle_path}/zh_test.pkl')

en_test_embeddings, en_test_labels = reshape_input(en_test_df['embeddings']), reshape_input(en_test_df['bin_label'])
fr_test_embeddings, fr_test_labels = reshape_input(fr_test_df['embeddings']), reshape_input(fr_test_df['bin_label'])
zh_test_embeddings, zh_test_labels = reshape_input(zh_test_df['embeddings']), reshape_input(zh_test_df['bin_label'])

results = collections.defaultdict(dict)
i = 1

for f in glob.glob(f'{params.classifier_path}/svm_classifier*.pickle'):
    print('>>> LOAD SVM %s'%f)
    svm_classifier = load_pickle(f)

    pred_y = svm_classifier.predict(en_test_embeddings)
    print('EN')
    report = classification_report(en_test_labels, pred_y, output_dict=True)
    print(pd.DataFrame(list(map(report.get, ['micro avg', 'macro avg', 'samples avg', 'weighted avg']))))
    results['EN'][i] = svm_classifier.score(en_test_embeddings, en_test_labels)

    pred_y = svm_classifier.predict(fr_test_embeddings)
    print('FR')
    report = classification_report(fr_test_labels, pred_y, output_dict=True)
    print(pd.DataFrame(list(map(report.get, ['micro avg', 'macro avg', 'samples avg', 'weighted avg']))))
    results['FR'][i] = svm_classifier.score(fr_test_embeddings, fr_test_labels)

    pred_y = svm_classifier.predict(zh_test_embeddings)
    print('ZH')
    report = classification_report(zh_test_labels, pred_y, output_dict=True)
    print(pd.DataFrame(list(map(report.get, ['micro avg', 'macro avg', 'samples avg', 'weighted avg']))))
    results['ZH'][i] = svm_classifier.score(zh_test_embeddings, zh_test_labels)

    pred_y = svm_classifier.predict( np.concatenate((en_test_embeddings, fr_test_embeddings, zh_test_embeddings)) )
    print('MIX')
    report = classification_report( np.concatenate((en_test_labels, fr_test_labels, zh_test_labels)), pred_y, output_dict=True )
    print(pd.DataFrame(list(map(report.get, ['micro avg', 'macro avg', 'samples avg', 'weighted avg']))))
    results['MIX'][i] = svm_classifier.score( np.concatenate((en_test_embeddings, fr_test_embeddings, zh_test_embeddings)) ,
                                              np.concatenate((en_test_labels, fr_test_labels, zh_test_labels)))

    i += 1

print(pd.DataFrame.from_dict(results, orient='index'))