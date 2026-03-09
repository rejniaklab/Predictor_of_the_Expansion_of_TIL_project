import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import StratifiedKFold, KFold

def MI_score(X,y):
    
    # Detect task type
    is_classification = y.nunique() < 20 and not np.issubdtype(y.dtype, np.floating)
    
    # Fill missing values (consistent across folds)
    X_filled = X.fillna(0)
    
    # Cross-validation
    n_splits = 5
    
    if is_classification:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
    mi_scores = []

    #
    for fold, (train_idx, _) in enumerate(cv.split(X, y)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        if is_classification:
            mi = mutual_info_classif(
                X_train,
                y_train,
                discrete_features='auto',
                random_state=42
            )
        else:
            mi = mutual_info_regression(
                X_train,
                y_train,
                random_state=42
            )

        mi_scores.append(mi)

    # Aggregate MI
    mi_scores = np.vstack(mi_scores)
    return mi_scores