import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def rf_importance(X_train, y_train, max_depth=10, class_weight=None, top_n=15, n_estimators=50, random_state=0):
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,
        class_weight=class_weight, n_jobs=-1
    )
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_labels = X_train.columns
    std = np.std(
        [tree.feature_importances_ for tree in model.estimators_], axis=0)

    for f in range(X_train.shape[1]):
        print(
            f"{f + 1}. feature no:{indices[f]} feature name:{feat_labels[indices[f]]} ({importances[indices[f]]:.6f})")

    indices = indices[:top_n]
    plt.figure()
    plt.title(f"Feature importances top {top_n}")
    plt.bar(range(top_n), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(top_n), indices)
    plt.xlim([-1, top_n])
    plt.show()
    return model


def gbt_importance(X_train, y_train, max_depth=10, top_n=15, n_estimators=50, random_state=0):
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_labels = X_train.columns
    std = np.std(
        [tree[0].feature_importances_ for tree in model.estimators_], axis=0)

    for f in range(X_train.shape[1]):
        print(
            f"{f + 1}. feature no:{indices[f]} feature name:{feat_labels[indices[f]]} ({importances[indices[f]]:.6f})")

    indices = indices[:top_n]
    plt.figure()
    plt.title(f"Feature importances top {top_n}")
    plt.bar(range(top_n), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(top_n), indices)
    plt.xlim([-1, top_n])
    plt.show()
    return model
