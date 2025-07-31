# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import warnings

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, offset=1):
        self.cols = cols
        self.offset = offset
        self.fitted_cols_ = []
        
    def fit(self, X, y=None):
        cols_to_transform = self.cols if self.cols else X.select_dtypes(include=[np.number]).columns
        
        for col in cols_to_transform:
            if col in X.columns and (X[col] > 0).all():
                self.fitted_cols_.append(col)
            elif col in X.columns:
                warnings.warn(f"Column {col} contains non-positive values, will add offset")
                self.fitted_cols_.append(col)
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in self.fitted_cols_:
            if col in X_transformed.columns:
                X_transformed[f"{col}_log"] = np.log(X_transformed[col] + self.offset)
        return X_transformed

class PowerTransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, method='yeo-johnson'):
        self.cols = cols
        self.method = method
        self.transformers_ = {}
        
    def fit(self, X, y=None):
        cols_to_transform = self.cols if self.cols else X.select_dtypes(include=[np.number]).columns
        
        for col in cols_to_transform:
            if col in X.columns:
                transformer = PowerTransformer(method=self.method, standardize=False)
                transformer.fit(X[[col]])
                self.transformers_[col] = transformer
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col, transformer in self.transformers_.items():
            if col in X_transformed.columns:
                X_transformed[f"{col}_{self.method}"] = transformer.transform(X_transformed[[col]]).flatten()
        return X_transformed

class QuantileTransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, output_distribution='normal', n_quantiles=1000):
        self.cols = cols
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles
        self.transformers_ = {}
        
    def fit(self, X, y=None):
        cols_to_transform = self.cols if self.cols else X.select_dtypes(include=[np.number]).columns
        
        for col in cols_to_transform:
            if col in X.columns:
                n_quantiles = min(self.n_quantiles, len(X[col].dropna()))
                transformer = QuantileTransformer(output_distribution=self.output_distribution, n_quantiles=n_quantiles)
                transformer.fit(X[[col]])
                self.transformers_[col] = transformer
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col, transformer in self.transformers_.items():
            if col in X_transformed.columns:
                X_transformed[f"{col}_quantile"] = transformer.transform(X_transformed[[col]]).flatten()
        return X_transformed

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, transform_func=None, func_name='custom'):
        self.cols = cols
        self.transform_func = transform_func
        self.func_name = func_name
        self.fitted_cols_ = []
        
    def fit(self, X, y=None):
        cols_to_transform = self.cols if self.cols else X.select_dtypes(include=[np.number]).columns
        self.fitted_cols_ = [col for col in cols_to_transform if col in X.columns]
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in self.fitted_cols_:
            if col in X_transformed.columns:
                try:
                    X_transformed[f"{col}_{self.func_name}"] = self.transform_func(X_transformed[col])
                except Exception as e:
                    warnings.warn(f"Transformation failed for column {col}: {e}")
        return X_transformed

class MultiTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformers_config):
        self.transformers_config = transformers_config
        self.transformers_ = {}
        
    def fit(self, X, y=None):
        for transformer_name, config in self.transformers_config.items():
            transformer_class = config['transformer']
            transformer_params = config.get('params', {})
            
            transformer = transformer_class(**transformer_params)
            transformer.fit(X, y)
            self.transformers_[transformer_name] = transformer
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for transformer_name, transformer in self.transformers_.items():
            X_transformed = transformer.transform(X_transformed)
        return X_transformed

def normality_test(data, column):
    stat, p_value = stats.shapiro(data[column].dropna())
    return {'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05}

def diagnostic_plots(df, variable, figsize=(12, 8)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    axes[0, 0].hist(df[variable].dropna(), bins=30, alpha=0.7)
    axes[0, 0].set_title(f'Histogram of {variable}')
    
    stats.probplot(df[variable].dropna(), dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title(f'Q-Q Plot of {variable}')
    
    axes[1, 0].boxplot(df[variable].dropna())
    axes[1, 0].set_title(f'Box Plot of {variable}')
    
    df[variable].plot(kind='density', ax=axes[1, 1])
    axes[1, 1].set_title(f'Density Plot of {variable}')
    
    plt.tight_layout()
    plt.show()
    
    normality = normality_test(df, variable)
    print(f"Normality Test Results for {variable}:")
    print(f"Shapiro-Wilk Statistic: {normality['statistic']:.4f}")
    print(f"P-value: {normality['p_value']:.4f}")
    print(f"Is Normal: {normality['is_normal']}")

def suggest_transformation(data, column):
    skewness = stats.skew(data[column].dropna())
    kurtosis = stats.kurtosis(data[column].dropna())
    
    suggestions = []
    
    if abs(skewness) > 1:
        if skewness > 0:
            suggestions.append("log transformation (right-skewed)")
            suggestions.append("square root transformation")
        else:
            suggestions.append("square transformation (left-skewed)")
    
    if abs(kurtosis) > 3:
        suggestions.append("Box-Cox transformation")
        suggestions.append("Quantile transformation")
    
    if not suggestions:
        suggestions.append("Data appears relatively normal")
    
    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'suggestions': suggestions
    }