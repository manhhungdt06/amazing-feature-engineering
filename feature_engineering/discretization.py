import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

class UniformDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5, strategy='uniform'):
        self.n_bins = n_bins
        self.strategy = strategy
        self.discretizers_ = {}
        
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.strategy)
            discretizer.fit(X[[col]])
            self.discretizers_[col] = discretizer
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col, discretizer in self.discretizers_.items():
            if col in X_transformed.columns:
                X_transformed[f"{col}_{self.strategy}"] = discretizer.transform(X_transformed[[col]]).flatten()
        return X_transformed

class DecisionTreeDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_depth=3, min_samples_leaf=50):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees_ = {}
        
    def fit(self, X, y):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if isinstance(self.max_depth, list):
                best_depth = self._optimize_depth(X[[col]], y, self.max_depth)
            else:
                best_depth = self.max_depth
                
            tree = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=self.min_samples_leaf, random_state=42)
            tree.fit(X[[col]], y)
            self.trees_[col] = tree
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col, tree in self.trees_.items():
            if col in X_transformed.columns:
                X_transformed[f"{col}_tree_bins"] = tree.apply(X_transformed[[col]])
        return X_transformed
    
    def _optimize_depth(self, X, y, depths):
        best_score = 0
        best_depth = depths[0]
        for depth in depths:
            tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=self.min_samples_leaf, random_state=42)
            scores = cross_val_score(tree, X, y, cv=3, scoring='roc_auc')
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_depth = depth
        return best_depth

class ChiMergeDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_bins=10, confidence_threshold=3.841, min_samples_per_bin=30):
        self.max_bins = max_bins
        self.confidence_threshold = confidence_threshold
        self.min_samples_per_bin = min_samples_per_bin
        self.bin_edges_ = {}
        
    def fit(self, X, y):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                bins = self._chimerge_binning(X[col], y)
                self.bin_edges_[col] = bins
            except Exception as e:
                warnings.warn(f"ChiMerge failed for column {col}: {e}")
                self.bin_edges_[col] = np.linspace(X[col].min(), X[col].max(), self.max_bins + 1)
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col, bins in self.bin_edges_.items():
            if col in X_transformed.columns:
                X_transformed[f"{col}_chimerge"] = pd.cut(X_transformed[col], bins=bins, include_lowest=True, duplicates='drop')
        return X_transformed
    
    def _chimerge_binning(self, feature, target):
        df = pd.DataFrame({'feature': feature, 'target': target}).dropna()
        df = df.sort_values('feature').reset_index(drop=True)
        
        unique_vals = df['feature'].unique()
        if len(unique_vals) <= self.max_bins:
            return np.concatenate([[df['feature'].min() - 0.001], unique_vals, [df['feature'].max() + 0.001]])
        
        intervals = []
        for val in unique_vals:
            subset = df[df['feature'] == val]
            pos_count = subset['target'].sum()
            neg_count = len(subset) - pos_count
            intervals.append([val, pos_count, neg_count])
        
        while len(intervals) > self.max_bins:
            chi_values = []
            for i in range(len(intervals) - 1):
                chi_val = self._calculate_chi_square(intervals[i], intervals[i + 1])
                chi_values.append((chi_val, i))
            
            if not chi_values:
                break
                
            min_chi, min_idx = min(chi_values)
            if min_chi >= self.confidence_threshold and len(intervals) <= self.max_bins:
                break
                
            intervals[min_idx][1] += intervals[min_idx + 1][1]
            intervals[min_idx][2] += intervals[min_idx + 1][2]
            intervals[min_idx][0] = intervals[min_idx + 1][0]
            intervals.pop(min_idx + 1)
        
        bins = [interval[0] for interval in intervals]
        bins = [df['feature'].min() - 0.001] + bins[1:] + [df['feature'].max() + 0.001]
        return np.array(sorted(set(bins)))
    
    def _calculate_chi_square(self, interval1, interval2):
        a, b = interval1[1], interval1[2]
        c, d = interval2[1], interval2[2]
        
        if (a + c) == 0 or (b + d) == 0 or (a + b) == 0 or (c + d) == 0:
            return 0
            
        numerator = ((a + b + c + d) * (a * d - b * c) ** 2)
        denominator = (a + b) * (c + d) * (a + c) * (b + d)
        
        return numerator / denominator if denominator > 0 else 0

class OptimalBinning(BaseEstimator, TransformerMixin):
    def __init__(self, method='uniform', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.binners_ = {}
        
    def fit(self, X, y=None):
        if self.method == 'uniform':
            self.binner = UniformDiscretizer(**self.kwargs)
        elif self.method == 'tree':
            self.binner = DecisionTreeDiscretizer(**self.kwargs)
        elif self.method == 'chimerge':
            self.binner = ChiMergeDiscretizer(**self.kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        self.binner.fit(X, y)
        return self
    
    def transform(self, X):
        return self.binner.transform(X)