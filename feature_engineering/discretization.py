import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
# from warnings import warn


class ChiMerge:
    def __init__(self, col=None, bins=None, confidenceVal=3.841, num_of_bins=10):
        self.col = col
        self._dim = None
        self.confidenceVal = confidenceVal
        self.bins = bins
        self.num_of_bins = num_of_bins

    def fit(self, X, y):
        self._dim = X.shape[1]
        _, bins = self.chimerge(X, y)
        self.bins = bins
        return self

    def transform(self, X):
        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')
        if X.shape[1] != self._dim:
            raise ValueError(
                f'Unexpected input dimension {X.shape[1]}, expected {self._dim}')
        X, _ = self.chimerge(X)
        return X

    def chimerge(self, X_in, y=None):
        X = X_in.copy()
        if self.bins is not None:
            X[self.col + '_chimerge'] = pd.cut(X[self.col],
                                               bins=self.bins, include_lowest=True)
        else:
            total_num = X.groupby([self.col])[y].count()
            positive_class = X.groupby([self.col])[y].sum()
            regroup = pd.merge(total_num, positive_class,
                               left_index=True, right_index=True)
            regroup['negative_class'] = regroup[0] - regroup[1]
            np_regroup = regroup.to_numpy()
            i = 0
            while i <= np_regroup.shape[0] - 2:
                if (np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or (np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0):
                    np_regroup[i, 1] += np_regroup[i + 1, 1]
                    np_regroup[i, 2] += np_regroup[i + 1, 2]
                    np_regroup[i, 0] = np_regroup[i + 1, 0]
                    np_regroup = np.delete(np_regroup, i + 1, 0)
                    i -= 1
                i += 1
            chi_table = np.array([(np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2 *
                                  (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) /
                                  ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) *
                                   (np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
                                  for i in range(np_regroup.shape[0] - 1)])
            while True:
                if len(chi_table) <= (self.num_of_bins - 1) and min(chi_table) >= self.confidenceVal:
                    break
                chi_min_index = np.argmin(chi_table)
                np_regroup[chi_min_index,
                           1] += np_regroup[chi_min_index + 1, 1]
                np_regroup[chi_min_index,
                           2] += np_regroup[chi_min_index + 1, 2]
                np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
                np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)
                chi_table = np.delete(chi_table, chi_min_index)
                if chi_min_index > 0:
                    chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 / \
                        ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) *
                         (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                if chi_min_index < np_regroup.shape[0] - 1:
                    chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - np_regroup[chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 / \
                        ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) *
                         (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
            result_data = pd.DataFrame(
                {'variable': [self.col] * np_regroup.shape[0]})
            result_data['interval'] = [
                f"{'-inf' if i == 0 else np_regroup[i - 1, 0]},{np_regroup[i, 0] if i < np_regroup.shape[0] - 1 else '+'}" for i in range(np_regroup.shape[0])]
            result_data['flag_0'] = np_regroup[:, 2]
            result_data['flag_1'] = np_regroup[:, 1]
            self.bins = sorted(X[self.col].min() - 0.1)
            return X, self.bins


class ChiMergeTodo():

    def __init__(self, col=None, bins=None, confidenceVal=3.841, num_of_bins=10, min_samples_per_bin=5):
        self.col = col
        self._dim = None
        self.confidenceVal = confidenceVal
        self.bins = bins
        self.num_of_bins = num_of_bins
        self.min_samples_per_bin = min_samples_per_bin

    def fit(self, X, y, **kwargs):
        self._dim = X.shape[1]
        _, bins = self.chimerge(X_in=X, y=y)
        self.bins = bins
        return self

    def transform(self, X):
        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (
                X.shape[1], self._dim,))
        X, _ = self.chimerge(X_in=X, bins=self.bins)
        return X

    def chimerge(self, X_in, y=None, bins=None):
        X = X_in.copy(deep=True)
        if bins is not None:
            try:
                X[self.col +
                    '_chimerge'] = pd.cut(X[self.col], bins=bins, include_lowest=True)
            except Exception as e:
                print(e)
        else:
            try:
                total_num = X.groupby([self.col])[y].count()
                positive_class = X.groupby([self.col])[y].sum()
                regroup = pd.merge(total_num, positive_class,
                                   left_index=True, right_index=True, how='inner')
                regroup.reset_index(inplace=True)
                regroup['negative_class'] = regroup['total_num'] - \
                    regroup['positive_class']
                regroup = regroup.drop('total_num', axis=1)
                np_regroup = np.array(regroup)

                i = 0
                while (i <= np_regroup.shape[0] - 2):
                    if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or (np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
                        np_regroup[i, 1] += np_regroup[i + 1, 1]
                        np_regroup[i, 2] += np_regroup[i + 1, 2]
                        np_regroup[i, 0] = np_regroup[i + 1, 0]
                        np_regroup = np.delete(np_regroup, i + 1, 0)
                        i -= 1
                    i += 1

                chi_table = np.array([])
                for i in np.arange(np_regroup.shape[0] - 1):
                    chi = (np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2 \
                        * (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) / \
                        ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) * (
                            np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
                    chi_table = np.append(chi_table, chi)

                while (1):
                    if (len(chi_table) <= (self.num_of_bins - 1) and min(chi_table) >= self.confidenceVal):
                        break
                    chi_min_index = np.argwhere(chi_table == min(chi_table))[0]

                    if np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1] >= self.min_samples_per_bin or \
                            np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2] >= self.min_samples_per_bin:

                        np_regroup[chi_min_index,
                                   1] += np_regroup[chi_min_index + 1, 1]
                        np_regroup[chi_min_index,
                                   2] += np_regroup[chi_min_index + 1, 2]
                        np_regroup[chi_min_index,
                                   0] = np_regroup[chi_min_index + 1, 0]
                        np_regroup = np.delete(
                            np_regroup, chi_min_index + 1, 0)

                    if (chi_min_index == np_regroup.shape[0] - 1):
                        chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                            * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                            ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                        chi_table = np.delete(chi_table, chi_min_index, axis=0)

                    else:
                        chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                            * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                            ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                        chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - np_regroup[chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 \
                            * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) / \
                            ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) * (
                                np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
                        chi_table = np.delete(
                            chi_table, chi_min_index + 1, axis=0)

                result_data = pd.DataFrame()
                result_data['variable'] = [self.col] * np_regroup.shape[0]
                bins = []
                tmp = []
                for i in np.arange(np_regroup.shape[0]):
                    if i == 0:
                        y = '-inf' + ',' + str(np_regroup[i, 0])
                    elif i == np_regroup.shape[0] - 1:
                        y = str(np_regroup[i - 1, 0]) + '+'
                    else:
                        y = str(np_regroup[i - 1, 0]) + \
                            ',' + str(np_regroup[i, 0])
                    bins.append(np_regroup[i - 1, 0])
                    tmp.append(y)

                bins.append(X[self.col].min() - 0.1)
                result_data['interval'] = tmp
                result_data['flag_0'] = np_regroup[:, 2]
                result_data['flag_1'] = np_regroup[:, 1]
                bins.sort(reverse=False)

            except Exception as e:
                print(e)

        return X, bins


class DiscretizeByDecisionTree():

    def __init__(self, col=None, max_depth=None, tree_model=None):
        self.col = col
        self._dim = None
        self.max_depth = max_depth
        self.tree_model = tree_model

    def fit(self, X, y, **kwargs):
        self._dim = X.shape[1]
        _, tree = self.discretize(
            X_in=X, y=y, max_depth=self.max_depth, col=self.col, tree_model=self.tree_model)
        self.tree_model = tree
        return self

    def transform(self, X):
        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (
                X.shape[1], self._dim,))
        X, _ = self.discretize(X_in=X, col=self.col,
                               tree_model=self.tree_model)
        return X

    def discretize(self, X_in, y=None, max_depth=None, tree_model=None, col=None):
        X = X_in.copy(deep=True)
        if tree_model is not None:
            X[col +
                '_tree_discret'] = tree_model.predict_proba(X[col].to_frame())[:, 1]
        else:
            if isinstance(max_depth, int):
                tree_model = DecisionTreeClassifier(max_depth=max_depth)
                tree_model.fit(X[col].to_frame(), y)
            elif len(max_depth) > 1:
                score_ls, score_std_ls = [], []
                for tree_depth in max_depth:
                    tree_model = DecisionTreeClassifier(max_depth=tree_depth)
                    scores = cross_val_score(
                        tree_model, X[col].to_frame(), y, cv=3, scoring='roc_auc')
                    score_ls.append(np.mean(scores))
                    score_std_ls.append(np.std(scores))
                temp = pd.DataFrame(
                    {'depth': max_depth, 'roc_auc_mean': score_ls, 'roc_auc_std': score_std_ls})
                optimal_depth = temp.loc[temp.roc_auc_mean.idxmax(), 'depth']
                tree_model = DecisionTreeClassifier(max_depth=optimal_depth)
                tree_model.fit(X[col].to_frame(), y)
            else:
                raise ValueError('max_depth must be an integer or a list')
        return X, tree_model
