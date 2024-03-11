from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from skopt import BayesSearchCV # Bayesian Optimisation for hyperparameters
from skopt.space import Real, Categorical, Integer

class SKClassifier():
    def __init__(self, **kwargs):
        self.train_x = kwargs['train_x']
        self.train_y = kwargs['train_y']

    def train(self, model_name, seed):

        classifier = MultiOutputClassifier(svm.SVC(random_state=seed))
        opt = BayesSearchCV(classifier,
                            {'estimator__C': Real(1e-6, 1e+6, prior='log-uniform'),
                             'estimator__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
                             'estimator__degree': Integer(1,8),
                             'estimator__kernel': Categorical(['linear', 'poly', 'rbf']),
                             },
                            n_iter=20,
                            random_state=seed,
                            cv=5,
                            scoring=make_scorer(f1_score, average='macro'))

        _ = opt.fit(self.train_x, self.train_y)

        return opt