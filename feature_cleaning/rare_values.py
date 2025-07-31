import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RareValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05, strategy='group', replacement='rare'):
        self.threshold = threshold
        self.strategy = strategy
        self.replacement = replacement
        self.mappings_ = {}
        
    def fit(self, X, y=None):
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            value_counts = X[col].value_counts(normalize=True)
            rare_values = value_counts[value_counts < self.threshold].index.tolist()
            
            if self.strategy == 'group':
                mapping = {val: self.replacement if val in rare_values else val 
                          for val in X[col].unique()}
            elif self.strategy == 'mode':
                mode_value = X[col].mode().iloc[0] if not X[col].mode().empty else X[col].iloc[0]
                mapping = {val: mode_value if val in rare_values else val 
                          for val in X[col].unique()}
            elif self.strategy == 'drop':
                mapping = rare_values
            
            self.mappings_[col] = {
                'mapping': mapping,
                'rare_values': rare_values,
                'strategy': self.strategy
            }
        
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        for col, info in self.mappings_.items():
            if col in X_transformed.columns:
                if info['strategy'] == 'drop':
                    X_transformed = X_transformed[~X_transformed[col].isin(info['rare_values'])]
                else:
                    X_transformed[col] = X_transformed[col].map(info['mapping']).fillna(X_transformed[col])
        
        return X_transformed
    
    def get_rare_summary(self):
        summary = []
        for col, info in self.mappings_.items():
            summary.append({
                'column': col,
                'rare_values': info['rare_values'],
                'rare_count': len(info['rare_values']),
                'strategy': info['strategy']
            })
        return pd.DataFrame(summary)

def detect_rare_values(data, threshold=0.05, return_summary=True):
    results = {}
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        value_counts = data[col].value_counts(normalize=True)
        rare_values = value_counts[value_counts < threshold]
        
        results[col] = {
            'rare_values': rare_values.index.tolist(),
            'rare_proportions': rare_values.to_dict(),
            'total_rare_count': len(rare_values),
            'rare_percentage': (rare_values.sum() * 100)
        }
    
    if return_summary:
        summary = pd.DataFrame([
            {
                'column': col,
                'rare_count': info['total_rare_count'],
                'rare_percentage': f"{info['rare_percentage']:.2f}%"
            }
            for col, info in results.items()
        ])
        return results, summary
    
    return results

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