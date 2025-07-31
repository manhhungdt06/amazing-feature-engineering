import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
# import pandas as pd
# import warnings

class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, smoothing=1.0, min_samples_leaf=1, noise_level=0.01):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.mappings_ = {}
        self.global_mean_ = None
        
    def fit(self, X, y):
        self.global_mean_ = y.mean()
        cols_to_encode = self.cols if self.cols else X.select_dtypes(include=['object', 'category']).columns
        
        for col in cols_to_encode:
            if col in X.columns:
                mapping = self._compute_mean_encoding(X[col], y)
                self.mappings_[col] = mapping
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col, mapping in self.mappings_.items():
            if col in X_transformed.columns:
                X_transformed[f"{col}_mean_encoded"] = X_transformed[col].map(mapping).fillna(self.global_mean_)
                if self.noise_level > 0:
                    noise = np.random.normal(0, self.noise_level, len(X_transformed))
                    X_transformed[f"{col}_mean_encoded"] += noise
        return X_transformed
    
    def _compute_mean_encoding(self, feature, target):
        stats = target.groupby(feature).agg(['count', 'mean'])
        stats.columns = ['count', 'mean']
        
        smoothed_means = (stats['count'] * stats['mean'] + self.smoothing * self.global_mean_) / (stats['count'] + self.smoothing)
        
        valid_categories = stats[stats['count'] >= self.min_samples_leaf].index
        mapping = smoothed_means.to_dict()
        
        for cat in stats.index:
            if cat not in valid_categories:
                mapping[cat] = self.global_mean_
                
        return mapping

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.frequency_maps_ = {}
        
    def fit(self, X, y=None):
        cols_to_encode = self.cols if self.cols else X.select_dtypes(include=['object', 'category']).columns
        
        for col in cols_to_encode:
            if col in X.columns:
                self.frequency_maps_[col] = X[col].value_counts(normalize=True).to_dict()
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col, freq_map in self.frequency_maps_.items():
            if col in X_transformed.columns:
                X_transformed[f"{col}_frequency"] = X_transformed[col].map(freq_map).fillna(0)
        return X_transformed

class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, drop_first=True):
        self.cols = cols
        self.drop_first = drop_first
        self.categories_ = {}
        
    def fit(self, X, y=None):
        cols_to_encode = self.cols if self.cols else X.select_dtypes(include=['object', 'category']).columns
        
        for col in cols_to_encode:
            if col in X.columns:
                self.categories_[col] = X[col].unique().tolist()
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        for col, categories in self.categories_.items():
            if col in X_transformed.columns:
                for i, category in enumerate(categories):
                    if self.drop_first and i == 0:
                        continue
                    X_transformed[f"{col}_{category}"] = (X_transformed[col] == category).astype(int)
                X_transformed.drop(col, axis=1, inplace=True)
        
        return X_transformed

class LabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.label_maps_ = {}
        
    def fit(self, X, y=None):
        cols_to_encode = self.cols if self.cols else X.select_dtypes(include=['object', 'category']).columns
        
        for col in cols_to_encode:
            if col in X.columns:
                unique_vals = X[col].unique()
                self.label_maps_[col] = {val: idx for idx, val in enumerate(unique_vals)}
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col, label_map in self.label_maps_.items():
            if col in X_transformed.columns:
                X_transformed[f"{col}_label_encoded"] = X_transformed[col].map(label_map).fillna(-1)
        return X_transformed

class MultiEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoders_config):
        self.encoders_config = encoders_config
        self.encoders_ = {}
        
    def fit(self, X, y=None):
        for encoder_name, config in self.encoders_config.items():
            encoder_class = config['encoder']
            encoder_params = config.get('params', {})
            encoder_cols = config.get('cols', None)
            
            encoder = encoder_class(cols=encoder_cols, **encoder_params)
            encoder.fit(X, y)
            self.encoders_[encoder_name] = encoder
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for encoder_name, encoder in self.encoders_.items():
            X_transformed = encoder.transform(X_transformed)
        return X_transformed