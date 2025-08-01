import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE, RFECV
import warnings

class RecursiveFeatureElimination(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 estimator=None,
                 step=1,
                 cv=3,
                 scoring='roc_auc',
                 min_features_to_select=1,
                 tolerance=0.001,
                 random_state=42):
        
        self.estimator = estimator
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.min_features_to_select = min_features_to_select
        self.tolerance = tolerance
        self.random_state = random_state
        
        self.selector_ = None
        self.selected_features_ = []
        self.elimination_history_ = []
        
    def fit(self, X, y):
        if self.estimator is None:
            if len(np.unique(y)) == 2:
                self.estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            else:
                self.estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        
        if self.tolerance > 0:
            self.selector_ = RFECV(
                estimator=self.estimator,
                step=self.step,
                cv=self.cv,
                scoring=self.scoring,
                min_features_to_select=self.min_features_to_select
            )
        else:
            n_features = max(self.min_features_to_select, len(X.columns) // 2)
            self.selector_ = RFE(
                estimator=self.estimator,
                n_features_to_select=n_features,
                step=self.step
            )
        
        self.selector_.fit(X, y)
        self.selected_features_ = X.columns[self.selector_.get_support()].tolist()
        
        if hasattr(self.selector_, 'cv_results_'):
            self.elimination_history_ = self.selector_.cv_results_
        
        print(f"RFE selected {len(self.selected_features_)} features")
        return self
    
    def transform(self, X):
        return X[self.selected_features_]
    
    def plot_cv_results(self, figsize=(10, 6)):
        if hasattr(self.selector_, 'grid_scores_'):
            scores = self.scores = self.selector_.grid_scores_
            plt.figure(figsize=figsize)
            plt.plot(range(1, len(scores) + 1), scores, marker='o')
            plt.xlabel('Number of Features')
            plt.ylabel(f'Cross-Validation Score ({self.scoring})')
            plt.title('Recursive Feature Elimination CV Results')
            plt.grid(True, alpha=0.3)
            plt.show()

class ForwardFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, 
                    estimator=None,
                    cv=3,
                    scoring='roc_auc',
                    max_features=None,
                    tolerance=0.001,
                    patience=5,
                    random_state=42):
        
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.max_features = max_features
        self.tolerance = tolerance
        self.patience = patience
        self.random_state = random_state
        
        self.selected_features_ = []
        self.selection_history_ = []
        self.best_score_ = -np.inf
        
    def fit(self, X, y):
        if self.estimator is None:
            if len(np.unique(y)) == 2:
                self.estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            else:
                self.estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        
        available_features = list(X.columns)
        current_features = []
        patience_counter = 0
        
        max_features = self.max_features or len(X.columns)
        
        while len(current_features) < max_features and available_features and patience_counter < self.patience:
            best_feature = None
            best_score = -np.inf
            
            for feature in available_features:
                candidate_features = current_features + [feature]
                
                try:
                    scores = cross_val_score(
                        self.estimator, 
                        X[candidate_features], 
                        y, 
                        cv=self.cv, 
                        scoring=self.scoring
                    )
                    score = np.mean(scores)
                    
                    if score > best_score:
                        best_score = score
                        best_feature = feature
                        
                except Exception as e:
                    warnings.warn(f"Error evaluating feature {feature}: {e}")
                    continue
            
            if best_feature and (best_score - self.best_score_) > self.tolerance:
                current_features.append(best_feature)
                available_features.remove(best_feature)
                self.best_score_ = best_score
                patience_counter = 0
                
                self.selection_history_.append({
                    'feature': best_feature,
                    'score': best_score,
                    'n_features': len(current_features)
                })
                
                print(f"Added feature {best_feature}, score: {best_score:.4f}")
            else:
                patience_counter += 1
                if best_feature:
                    print(f"Feature {best_feature} did not improve score enough (improvement: {best_score - self.best_score_:.4f})")
        
        self.selected_features_ = current_features
        print(f"Forward selection completed: {len(self.selected_features_)} features selected")
        return self
    
    def transform(self, X):
        return X[self.selected_features_]

class BackwardFeatureElimination(BaseEstimator, TransformerMixin):
    def __init__(self, 
                    estimator=None,
                    cv=3,
                    scoring='roc_auc',
                    min_features=1,
                    tolerance=0.001,
                    patience=5,
                    random_state=42):
        
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.min_features = min_features
        self.tolerance = tolerance
        self.patience = patience
        self.random_state = random_state
        
        self.selected_features_ = []
        self.elimination_history_ = []
        self.best_score_ = -np.inf
        
    def fit(self, X, y):
        if self.estimator is None:
            if len(np.unique(y)) == 2:
                self.estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            else:
                self.estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        
        current_features = list(X.columns)
        
        initial_scores = cross_val_score(self.estimator, X[current_features], y, cv=self.cv, scoring=self.scoring)
        self.best_score_ = np.mean(initial_scores)
        
        patience_counter = 0
        
        while len(current_features) > self.min_features and patience_counter < self.patience:
            worst_feature = None
            best_score_after_removal = -np.inf
            
            for feature in current_features:
                candidate_features = [f for f in current_features if f != feature]
                
                try:
                    scores = cross_val_score(
                        self.estimator, 
                        X[candidate_features], 
                        y, 
                        cv=self.cv, 
                        scoring=self.scoring
                    )
                    score = np.mean(scores)
                    
                    if score > best_score_after_removal:
                        best_score_after_removal = score
                        worst_feature = feature
                        
                except Exception as e:
                    warnings.warn(f"Error evaluating removal of feature {feature}: {e}")
                    continue
            
            score_drop = self.best_score_ - best_score_after_removal
            
            if worst_feature and score_drop <= self.tolerance:
                current_features.remove(worst_feature)
                self.best_score_ = best_score_after_removal
                patience_counter = 0
                
                self.elimination_history_.append({
                    'removed_feature': worst_feature,
                    'score_after_removal': best_score_after_removal,
                    'score_drop': score_drop,
                    'n_features': len(current_features)
                })
                
                print(f"Removed feature {worst_feature}, score: {best_score_after_removal:.4f}, drop: {score_drop:.4f}")
            else:
                patience_counter += 1
                if worst_feature:
                    print(f"Stopping: removing {worst_feature} would drop score by {score_drop:.4f}")
        
        self.selected_features_ = current_features
        print(f"Backward elimination completed: {len(self.selected_features_)} features selected")
        return self
    
    def transform(self, X):
        return X[self.selected_features_]

class StepwiseSelection(BaseEstimator, TransformerMixin):
    def __init__(self, 
                    estimator=None,
                    cv=3,
                    scoring='roc_auc',
                    forward_threshold=0.01,
                    backward_threshold=0.01,
                    max_iter=100,
                    random_state=42):
        
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.forward_threshold = forward_threshold
        self.backward_threshold = backward_threshold
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.selected_features_ = []
        self.stepwise_history_ = []
        
    def fit(self, X, y):
        if self.estimator is None:
            if len(np.unique(y)) == 2:
                self.estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            else:
                self.estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        
        available_features = list(X.columns)
        current_features = []
        
        for iteration in range(self.max_iter):
            changed = False
            
            # Forward step
            if available_features:
                best_forward_feature = None
                best_forward_score = -np.inf
                
                current_score = self._get_score(X[current_features], y) if current_features else -np.inf
                
                for feature in available_features:
                    candidate_features = current_features + [feature]
                    score = self._get_score(X[candidate_features], y)
                    
                    if score > best_forward_score:
                        best_forward_score = score
                        best_forward_feature = feature
                
                if best_forward_feature and (best_forward_score - current_score) > self.forward_threshold:
                    current_features.append(best_forward_feature)
                    available_features.remove(best_forward_feature)
                    changed = True
                    
                    self.stepwise_history_.append({
                        'step': 'forward',
                        'feature': best_forward_feature,
                        'score': best_forward_score,
                        'iteration': iteration
                    })
            
            # Backward step
            if len(current_features) > 1:
                worst_backward_feature = None
                best_backward_score = -np.inf
                
                current_score = self._get_score(X[current_features], y)
                
                for feature in current_features:
                    candidate_features = [f for f in current_features if f != feature]
                    score = self._get_score(X[candidate_features], y)
                    
                    if score > best_backward_score:
                        best_backward_score = score
                        worst_backward_feature = feature
                
                score_drop = current_score - best_backward_score
                
                if worst_backward_feature and score_drop <= self.backward_threshold:
                    current_features.remove(worst_backward_feature)
                    available_features.append(worst_backward_feature)
                    changed = True
                    
                    self.stepwise_history_.append({
                        'step': 'backward',
                        'feature': worst_backward_feature,
                        'score': best_backward_score,
                        'iteration': iteration
                    })
            
            if not changed:
                print(f"Stepwise selection converged at iteration {iteration}")
                break
        
        self.selected_features_ = current_features
        print(f"Stepwise selection completed: {len(self.selected_features_)} features selected")
        return self
    
    def transform(self, X):
        return X[self.selected_features_]
    
    def _get_score(self, X, y):
        if len(X.columns) == 0:
            return -np.inf
        try:
            scores = cross_val_score(self.estimator, X, y, cv=self.cv, scoring=self.scoring)
            return np.mean(scores)
        except:
            return -np.inf