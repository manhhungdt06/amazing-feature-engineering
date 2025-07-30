from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def recursive_feature_elimination_rf(X_train, y_train, X_test, y_test, tol=0.001,
                                     max_depth=None, class_weight=None,
                                     n_estimators=50, random_state=0):
    features_to_remove, count = [], 1
    model_all = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=random_state, class_weight=class_weight,
                                       n_jobs=-1)
    model_all.fit(X_train, y_train)
    auc_all = roc_auc_score(y_test, model_all.predict_proba(X_test)[:, 1])

    for feature in X_train.columns:
        print(f"\nTesting feature: {feature} ({count}/{len(X_train.columns)})")
        count += 1
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=random_state, class_weight=class_weight,
                                       n_jobs=-1)
        model.fit(X_train.drop(features_to_remove +
                  [feature], axis=1), y_train)
        auc_int = roc_auc_score(y_test, model.predict_proba(
            X_test.drop(features_to_remove + [feature], axis=1))[:, 1])

        print(
            f"New Test ROC AUC={auc_int}\nAll features Test ROC AUC={auc_all}")
        diff_auc = auc_all - auc_int

        if diff_auc >= tol:
            print(f"Drop in ROC AUC={diff_auc} | keep: {feature}")
        else:
            print(f"Drop in ROC AUC={diff_auc} | remove: {feature}")
            auc_all = auc_int
            features_to_remove.append(feature)

    print(f"DONE!!\nTotal features to remove: {len(features_to_remove)}")
    features_to_keep = [
        x for x in X_train.columns if x not in features_to_remove]
    print(f"Total features to keep: {len(features_to_keep)}")

    return features_to_keep


def recursive_feature_addition_rf(X_train, y_train, X_test, y_test, tol=0.001,
                                  max_depth=None, class_weight=None,
                                  n_estimators=50, random_state=0):
    features_to_keep = [X_train.columns[0]]
    count = 1
    model_initial = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           random_state=random_state, class_weight=class_weight,
                                           n_jobs=-1)
    model_initial.fit(X_train[[X_train.columns[0]]], y_train)
    auc_all = roc_auc_score(y_test, model_initial.predict_proba(
        X_test[[X_train.columns[0]]])[:, 1])

    for feature in X_train.columns[1:]:
        print(f"\nTesting feature: {feature} ({count}/{len(X_train.columns)})")
        count += 1
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=random_state, class_weight=class_weight,
                                       n_jobs=-1)
        model.fit(X_train[features_to_keep + [feature]], y_train)
        auc_int = roc_auc_score(y_test, model.predict_proba(
            X_test[features_to_keep + [feature]])[:, 1])

        print(
            f"New Test ROC AUC={auc_int}\nAll features Test ROC AUC={auc_all}")
        diff_auc = auc_int - auc_all

        if diff_auc >= tol:
            print(f"Increase in ROC AUC={diff_auc} | keep: {feature}")
            auc_all = auc_int
            features_to_keep.append(feature)
        else:
            print(f"Increase in ROC AUC={diff_auc} | remove: {feature}")

    print(f"DONE!!\nTotal features to keep: {len(features_to_keep)}")

    return features_to_keep
