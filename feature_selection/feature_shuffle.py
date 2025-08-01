import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
import warnings

class PermutationImportanceSelector(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 estimator=None,
                 scoring='roc_auc',
                 n_repeats=5,
                 random_state=42,
                 threshold=0.001,
                 cv=None):
        
        self.estimator = estimator
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.threshold = threshold
        self.cv = cv
        
        self.feature_importances_ = None
        self.baseline_score_ = None
        self.selected_features_ = []
        
    def fit(self, X, y):
        if self.estimator is None:
            if len(np.unique(y)) == 2:
                self.estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            else:
                self.estimator = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        
        self.estimator.fit(X, y)
        
        if self.cv:
            baseline_scores = cross_val_score(self.estimator, X, y, cv=self.cv, scoring=self.scoring)
            self.baseline_score_ = np.mean(baseline_scores)
        else:
            self.baseline_score_ = self._calculate_score(X, y)
        
        feature_importances = {}
        
        for feature in X.columns:
            importance_values = []
            
            for repeat in range(self.n_repeats):
                X_permuted = X.copy()
                np.random.seed(self.random_state + repeat)
                X_permuted[feature] = np.random.permutation(X_permuted[feature])
                
                if self.cv:
                    permuted_scores = cross_val_score(self.estimator, X_permuted, y, cv=self.cv, scoring=self.scoring)
                    permuted_score = np.mean(permuted_scores)
                else:
                    permuted_score = self._calculate_score(X_permuted, y)
                
                importance = self.baseline_score_ - permuted_score
                importance_values.append(importance)
            
            feature_importances[feature] = {
                'mean_importance': np.mean(importance_values),
                'std_importance': np.std(importance_values),
                'importance_values': importance_values
            }
        
        self.feature_importances_ = pd.DataFrame(feature_importances).T
        self.feature_importances_ = self.feature_importances_.sort_values('mean_importance', ascending=False)
        
        self.selected_features_ = self.feature_importances_[
            self.feature_importances_['mean_importance'] > self.threshold
        ].index.tolist()
        
        print(f"Permutation importance: {len(self.selected_features_)} features selected")
        return self
    
    def transform(self, X):
        return X[self.selected_features_]
    
    def _calculate_score(self, X, y):
        if self.scoring == 'roc_auc':
            y_pred_proba = self.estimator.predict_proba(X)[:, 1]
            return roc_auc_score(y, y_pred_proba)
        elif self.scoring == 'accuracy':
            y_pred = self.estimator.predict(X)
            return accuracy_score(y, y_pred)
        elif self.scoring == 'neg_mean_squared_error':
            y_pred = self.estimator.predict(X)
            return -mean_squared_error(y, y_pred)
    
    def plot_importance(self, top_n=20, figsize=(12, 8)):
        top_features = self.feature_importances_.head(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_features)), top_features['mean_importance'], 
                xerr=top_features['std_importance'], alpha=0.7)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Permutation Importance')
        plt.title(f'Top {top_n} Features by Permutation Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

class DroppingFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 estimator=None,
                 scoring='roc_auc',
                 cv=3,
                 threshold=0.001,
                 random_state=42):
        
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.threshold = threshold
        self.random_state = random_state
        
        self.feature_importances_ = None
        self.baseline_score_ = None
        self.selected_features_ = []
        
    def fit(self, X, y):
        if self.estimator is None:
            if len(np.unique(y)) == 2:
                self.estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            else:
                self.estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        
        baseline_scores = cross_val_score(self.estimator, X, y, cv=self.cv, scoring=self.scoring)
        self.baseline_score_ = np.mean(baseline_scores)
        
        feature_importances = {}
        
        for feature in X.columns:
            X_dropped = X.drop(columns=[feature])
            
            try:
                dropped_scores = cross_val_score(self.estimator, X_dropped, y, cv=self.cv, scoring=self.scoring)
                dropped_score = np.mean(dropped_scores)
                importance = self.baseline_score_ - dropped_score
                feature_importances[feature] = importance
            except Exception as e:
                warnings.warn(f"Error processing feature {feature}: {e}")
                feature_importances[feature] = 0
        
        self.feature_importances_ = pd.Series(feature_importances).sort_values(ascending=False)
        
        self.selected_features_ = self.feature_importances_[
            self.feature_importances_ > self.threshold
        ].index.tolist()
        
        print(f"Drop feature importance: {len(self.selected_features_)} features selected")
        return self
    
    def transform(self, X):
        return X[self.selected_features_]