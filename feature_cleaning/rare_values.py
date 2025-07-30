import pandas as pd
# import numpy as np


class GroupingRareValues():
    def __init__(self, mapping=None, cols=None, threshold=0.01):
        self.cols = cols
        self.mapping = mapping
        self._dim = None
        self.threshold = threshold

    def fit(self, X, y=None, **kwargs):
        self._dim = X.shape[1]
        _, categories = self.grouping(
            X, mapping=self.mapping, cols=self.cols, threshold=self.threshold)
        self.mapping = categories
        return self

    def transform(self, X):
        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')

        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (
                X.shape[1], self._dim,))

        X, _ = self.grouping(X, mapping=self.mapping,
                             cols=self.cols, threshold=self.threshold)
        return X

    def grouping(self, X_in, threshold, mapping=None, cols=None):
        X = X_in.copy(deep=True)

        if mapping is not None:
            mapping_out = mapping
            for i in mapping:
                column = i.get('col')
                X[column] = X[column].map(i['mapping'])
        else:
            mapping_out = []
            for col in cols:
                temp_df = pd.Series(X[col].value_counts()/len(X))
                mapping = {k: (
                    'rare' if k not in temp_df[temp_df >= threshold].index else k) for k in temp_df.index}
                mapping = pd.Series(mapping)
                mapping_out.append(
                    {'col': col, 'mapping': mapping, 'data_type': X[col].dtype})

        return X, mapping_out


class ModeImputation():
    def __init__(self, mapping=None, cols=None, threshold=0.01):
        self.cols = cols
        self.mapping = mapping
        self._dim = None
        self.threshold = threshold

    def fit(self, X, y=None, **kwargs):
        self._dim = X.shape[1]
        _, categories = self.impute_with_mode(
            X, mapping=self.mapping, cols=self.cols, threshold=self.threshold)
        self.mapping = categories
        return self

    def transform(self, X):
        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')

        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (
                X.shape[1], self._dim,))

        X, _ = self.impute_with_mode(
            X, mapping=self.mapping, cols=self.cols, threshold=self.threshold)
        return X

    def impute_with_mode(self, X_in, threshold, mapping=None, cols=None):
        X = X_in.copy(deep=True)

        if mapping is not None:
            mapping_out = mapping
            for i in mapping:
                column = i.get('col')
                X[column] = X[column].map(i['mapping'])
        else:
            mapping_out = []
            for col in cols:
                temp_df = pd.Series(X[col].value_counts()/len(X))
                median = X[col].mode()[0]
                mapping = {k: (
                    median if k not in temp_df[temp_df >= threshold].index else k) for k in temp_df.index}
                mapping = pd.Series(mapping)
                mapping_out.append(
                    {'col': col, 'mapping': mapping, 'data_type': X[col].dtype})

        return X, mapping_out


# ==============================================================================
# def rare_imputation(X_train, X_test, variable):

#     # find the most frequent category
#     # frequent_cat = X_train.groupby(variable)[variable].count().sort_values().tail(1).index.values[0]
#     frequent_cat = X_train[variable].mode().iloc[0]

#     # find rare labels
#     temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))
#     rare_cat = [x for x in temp.loc[temp<0.05].index.values]

#     # create new variables, with Rare labels imputed

#     # by the most frequent category
#     X_train[variable+'_freq_imp'] = np.where(X_train[variable].isin(rare_cat), frequent_cat, X_train[variable])
#     X_test[variable+'_freq_imp'] = np.where(X_test[variable].isin(rare_cat), frequent_cat, X_test[variable])

#     # by adding a new label 'Rare'
#     X_train[variable+'_rare_imp'] = np.where(X_train[variable].isin(rare_cat), 'Rare', X_train[variable])
#     X_test[variable+'_rare_imp'] = np.where(X_test[variable].isin(rare_cat), 'Rare', X_test[variable])
#     return X_train, X_test
