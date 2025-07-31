import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.model_selection import cross_val_score
# import warnings

class TreeBasedSelector(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 estimator='random_forest',
                 task='classification',
                 n_estimators=100,
                 max_depth=None,
                 random_state=42,
                 threshold='mean',
                 top_k=None):
        
        self.estimator = estimator
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.threshold = threshold
        self.top_k = top_k
        
        self.model_ = None
        self.selector_ = None
        self.feature_importances_ = None
        
    def fit(self, X, y):
        if self.task == 'classification':
            if self.estimator == 'random_forest':
                self.model_ = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            elif self.estimator == 'gradient_boosting':
                self.model_ = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state
                )
        else:
            if self.estimator == 'random_forest':
                self.model_ = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                )
        
        self.model_.fit(X, y)
        
        self.feature_importances_ = pd.Series(
            self.model_.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        if self.top_k:
            selected_features = self.feature_importances_.head(self.top_k).index
            self.selector_ = selected_features
        else:
            self.selector_ = SelectFromModel(self.model_, threshold=self.threshold, prefit=True)
        
        return self
    
    def transform(self, X):
        if self.top_k:
            return X[self.selector_]
        else:
            return pd.DataFrame(
                self.selector_.transform(X),
                columns=X.columns[self.selector_.get_support()],
                index=X.index
            )
    
    def plot_importance(self, top_n=20, figsize=(10, 8)):
        top_features = self.feature_importances_.head(top_n)
        
        plt.figure(figsize=figsize)
        top_features.plot(kind='barh')
        plt.title(f'Top {top_n} Feature Importances ({self.estimator})')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance_df(self):
        return self.feature_importances_.reset_index().rename(
            columns={'index': 'feature', 0: 'importance'}
        )

class LinearSelector(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 estimator='lasso',
                 task='classification',
                 alpha=None,
                 cv=5,
                 random_state=42,
                 threshold='mean'):
        
        self.estimator = estimator
        self.task = task
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.threshold = threshold
        
        self.model_ = None
        self.selector_ = None
        self.coefficients_ = None
        
    def fit(self, X, y):
        if self.task == 'classification':
            if self.estimator == 'lasso':
                self.model_ = LogisticRegressionCV(
                    penalty='l1',
                    solver='liblinear',
                    cv=self.cv,
                    random_state=self.random_state,
                    max_iter=1000
                )
            elif self.estimator == 'ridge':
                self.model_ = LogisticRegressionCV(
                    penalty='l2',
                    cv=self.cv,
                    random_state=self.random_state,
                    max_iter=1000
                )
        else:
            if self.estimator == 'lasso':
                self.model_ = LassoCV(cv=self.cv, random_state=self.random_state)
            elif self.estimator == 'ridge':
                self.model_ = RidgeCV(cv=self.cv)
            elif self.estimator == 'elastic_net':
                self.model_ = ElasticNetCV(cv=self.cv, random_state=self.random_state)
        
        self.model_.fit(X, y)
        
        if hasattr(self.model_, 'coef_'):
            if self.model_.coef_.ndim > 1:
                coefficients = self.model_.coef_[0]
            else:
                coefficients = self.model_.coef_
        else:
            coefficients = self.model_.feature_importances_
        
        self.coefficients_ = pd.Series(
            np.abs(coefficients),
            index=X.columns
        ).sort_values(ascending=False)
        
        self.selector_ = SelectFromModel(self.model_, threshold=self.threshold, prefit=True)
        
        return self
    
    def transform(self, X):
        return pd.DataFrame(
            self.selector_.transform(X),
            columns=X.columns[self.selector_.get_support()],
            index=X.index
        )
    
    def get_selected_features(self):
        return X.columns[self.selector_.get_support()].tolist()

class EnsembleSelector(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 selectors_config,
                 voting_threshold=0.5,
                 combination_method='union'):
        
        self.selectors_config = selectors_config
        self.voting_threshold = voting_threshold
        self.combination_method = combination_method
        
        self.selectors_ = {}
        self.feature_votes_ = None
        self.selected_features_ = []
        
    def fit(self, X, y):
        feature_selections = {}
        
        for name, config in self.selectors_config.items():
            selector_class = config['selector']
            selector_params = config.get('params', {})
            
            selector = selector_class(**selector_params)
            selector.fit(X, y)
            
            if hasattr(selector, 'get_selected_features'):
                selected = selector.get_selected_features()
            else:
                selected = selector.transform(X).columns.tolist()
            
            feature_selections[name] = selected
            self.selectors_[name] = selector
        
        all_features = set()
        for features in feature_selections.values():
            all_features.update(features)
        
        feature_votes = {}
        for feature in all_features:
            votes = sum(1 for features in feature_selections.values() if feature in features)
            feature_votes[feature] = votes / len(self.selectors_config)
        
        self.feature_votes_ = pd.Series(feature_votes).sort_values(ascending=False)
        
        if self.combination_method == 'union':
            self.selected_features_ = list(all_features)
        elif self.combination_method == 'intersection':
            self.selected_features_ = list(set.intersection(*[set(features) for features in feature_selections.values()]))
        elif self.combination_method == 'voting':
            self.selected_features_ = self.feature_votes_[self.feature_votes_ >= self.voting_threshold].index.tolist()
        
        print(f"Ensemble selection: {len(self.selected_features_)} features selected")
        return self
    
    def transform(self, X):
        return X[self.selected_features_]
    
    def get_voting_results(self):
        return self.feature_votes_.reset_index().rename(
            columns={'index': 'feature', 0: 'vote_percentage'}
        )