import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest, SelectPercentile, f_classif, f_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import pearsonr    # , spearmanr
import warnings

class ConstantFeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.98):
        self.threshold = threshold
        self.constant_features_ = []
        
    def fit(self, X, y=None):
        self.constant_features_ = []
        for feature in X.columns:
            if X[feature].dtype in ['object', 'category']:
                max_freq = X[feature].value_counts(normalize=True).max()
            else:
                max_freq = (X[feature].value_counts(normalize=True)).max()
            
            if max_freq >= self.threshold:
                self.constant_features_.append(feature)
        
        print(f"{len(self.constant_features_)} constant features detected")
        return self
    
    def transform(self, X):
        return X.drop(columns=self.constant_features_, errors='ignore')
    
    def get_constant_features(self):
        return self.constant_features_

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95, method='pearson'):
        self.threshold = threshold
        self.method = method
        self.correlated_features_ = []
        self.correlation_groups_ = []
        
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if self.method == 'pearson':
            corr_matrix = X[numeric_cols].corr().abs()
        elif self.method == 'spearman':
            corr_matrix = X[numeric_cols].corr(method='spearman').abs()
        
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        corr_pairs = []
        for col in upper_tri.columns:
            for row in upper_tri.index:
                if pd.notna(upper_tri.loc[row, col]) and upper_tri.loc[row, col] >= self.threshold:
                    corr_pairs.append((row, col, upper_tri.loc[row, col]))
        
        corr_df = pd.DataFrame(corr_pairs, columns=['feature1', 'feature2', 'correlation'])
        corr_df = corr_df.sort_values('correlation', ascending=False)
        
        self.correlation_groups_ = []
        processed_features = set()
        
        for _, row in corr_df.iterrows():
            f1, f2 = row['feature1'], row['feature2']
            if f1 not in processed_features and f2 not in processed_features:
                group = self._find_correlation_group(corr_df, f1, processed_features)
                if len(group) > 1:
                    self.correlation_groups_.append(group)
                    processed_features.update(group)
        
        for group in self.correlation_groups_:
            self.correlated_features_.extend(group[1:])
        
        print(f"{len(self.correlated_features_)} correlated features detected")
        return self
    
    def transform(self, X):
        return X.drop(columns=self.correlated_features_, errors='ignore')
    
    def _find_correlation_group(self, corr_df, start_feature, processed):
        group = {start_feature}
        queue = [start_feature]
        
        while queue:
            current = queue.pop(0)
            related = corr_df[
                (corr_df['feature1'] == current) | (corr_df['feature2'] == current)
            ]
            
            for _, row in related.iterrows():
                f1, f2 = row['feature1'], row['feature2']
                new_feature = f2 if f1 == current else f1
                
                if new_feature not in group and new_feature not in processed:
                    group.add(new_feature)
                    queue.append(new_feature)
        
        return list(group)

class StatisticalFilter(BaseEstimator, TransformerMixin):
    def __init__(self, method='mutual_info', k=10, score_func=None):
        self.method = method
        self.k = k
        self.score_func = score_func
        self.selected_features_ = []
        self.scores_ = None
        
    def fit(self, X, y):
        if self.method == 'mutual_info':
            score_func = mutual_info_classif
        elif self.method == 'chi2':
            score_func = chi2
        elif self.method == 'f_classif':
            score_func = f_classif
        elif self.method == 'f_regression':
            score_func = f_regression
        elif self.method == 'custom' and self.score_func:
            score_func = self.score_func
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if isinstance(self.k, float) and 0 < self.k < 1:
            selector = SelectPercentile(score_func, percentile=self.k * 100)
        else:
            selector = SelectKBest(score_func, k=self.k)
        
        selector.fit(X, y)
        self.selected_features_ = X.columns[selector.get_support()].tolist()
        self.scores_ = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
        
        return self
    
    def transform(self, X):
        return X[self.selected_features_]

class UnivariateFilter(BaseEstimator, TransformerMixin):
    def __init__(self, method='roc_auc', threshold=0.6, task='classification'):
        self.method = method
        self.threshold = threshold
        self.task = task
        self.feature_scores_ = None
        self.selected_features_ = []
        
    def fit(self, X, y):
        scores = {}
        
        for feature in X.columns:
            try:
                if self.task == 'classification':
                    if self.method == 'roc_auc':
                        model = DecisionTreeClassifier(random_state=42, max_depth=3)
                        model.fit(X[[feature]], y)
                        y_pred_proba = model.predict_proba(X[[feature]])[:, 1]
                        score = roc_auc_score(y, y_pred_proba)
                    elif self.method == 'correlation':
                        score = abs(pearsonr(X[feature], y)[0])
                else:
                    if self.method == 'mse':
                        model = DecisionTreeRegressor(random_state=42, max_depth=3)
                        model.fit(X[[feature]], y)
                        y_pred = model.predict(X[[feature]])
                        score = -mean_squared_error(y, y_pred)
                    elif self.method == 'correlation':
                        score = abs(pearsonr(X[feature], y)[0])
                
                scores[feature] = score
                
            except Exception as e:
                warnings.warn(f"Error processing feature {feature}: {e}")
                scores[feature] = 0
        
        self.feature_scores_ = pd.Series(scores).sort_values(ascending=False)
        
        if self.task == 'classification' and self.method == 'roc_auc':
            self.selected_features_ = self.feature_scores_[self.feature_scores_ > self.threshold].index.tolist()
        elif self.method == 'correlation':
            self.selected_features_ = self.feature_scores_[self.feature_scores_ > self.threshold].index.tolist()
        else:
            self.selected_features_ = self.feature_scores_[self.feature_scores_ > self.threshold].index.tolist()
        
        print(f"{len(self.selected_features_)} features selected based on {self.method}")
        return self
    
    def transform(self, X):
        return X[self.selected_features_]

class VarianceFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.low_variance_features_ = []
        
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if X[col].var() <= self.threshold:
                self.low_variance_features_.append(col)
        
        print(f"{len(self.low_variance_features_)} low variance features detected")
        return self
    
    def transform(self, X):
        return X.drop(columns=self.low_variance_features_, errors='ignore')

class ComprehensiveFilter(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 constant_threshold=0.98,
                 correlation_threshold=0.95,
                 variance_threshold=0.0,
                 statistical_method='mutual_info',
                 statistical_k=20,
                 univariate_method='roc_auc',
                 univariate_threshold=0.6):
        
        self.constant_filter = ConstantFeatureFilter(constant_threshold)
        self.correlation_filter = CorrelationFilter(correlation_threshold)
        self.variance_filter = VarianceFilter(variance_threshold)
        self.statistical_filter = StatisticalFilter(statistical_method, statistical_k)
        self.univariate_filter = UnivariateFilter(univariate_method, univariate_threshold)
        
        self.filtering_results_ = {}
        
    def fit(self, X, y):
        X_temp = X.copy()
        
        print("Step 1: Removing constant features...")
        self.constant_filter.fit(X_temp)
        X_temp = self.constant_filter.transform(X_temp)
        self.filtering_results_['constant'] = len(X.columns) - len(X_temp.columns)
        
        print("Step 2: Removing low variance features...")
        self.variance_filter.fit(X_temp)
        X_temp = self.variance_filter.transform(X_temp)
        self.filtering_results_['variance'] = len(X.columns) - len(X_temp.columns) - self.filtering_results_['constant']
        
        print("Step 3: Removing correlated features...")
        self.correlation_filter.fit(X_temp)
        X_temp = self.correlation_filter.transform(X_temp)
        self.filtering_results_['correlation'] = len(X.columns) - len(X_temp.columns) - sum(self.filtering_results_.values())
        
        print("Step 4: Statistical feature selection...")
        if len(X_temp.columns) > 0:
            self.statistical_filter.fit(X_temp, y)
            X_temp = self.statistical_filter.transform(X_temp)
        self.filtering_results_['statistical'] = len(X.columns) - len(X_temp.columns) - sum(self.filtering_results_.values())
        
        print("Step 5: Univariate feature selection...")
        if len(X_temp.columns) > 0:
            self.univariate_filter.fit(X_temp, y)
            X_temp = self.univariate_filter.transform(X_temp)
        self.filtering_results_['univariate'] = len(X.columns) - len(X_temp.columns) - sum(self.filtering_results_.values())
        
        print(f"\nFiltering Summary:")
        print(f"Original features: {len(X.columns)}")
        print(f"Final features: {len(X_temp.columns)}")
        for step, removed in self.filtering_results_.items():
            print(f"{step.capitalize()} removed: {removed}")
        
        return self
    
    def transform(self, X):
        X_temp = X.copy()
        X_temp = self.constant_filter.transform(X_temp)
        X_temp = self.variance_filter.transform(X_temp)
        X_temp = self.correlation_filter.transform(X_temp)
        
        if len(X_temp.columns) > 0:
            X_temp = self.statistical_filter.transform(X_temp)
        if len(X_temp.columns) > 0:
            X_temp = self.univariate_filter.transform(X_temp)
        
        return X_temp