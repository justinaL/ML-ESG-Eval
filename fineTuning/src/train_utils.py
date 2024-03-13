from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import EvalPrediction
import torch
import numpy as np


def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    ## compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

    precision = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    ## return as dictionary
    metrics = {'f1_micro': f1_micro_average, 'f1_macro' : f1_macro_average, 'f1_weigthed' : f1_weighted_average,
               'precision' : precision, 'recall' : recall, 'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result
