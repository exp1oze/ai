from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

@dataclass
class ModelEvaluationResult:
    name: str
    accuracy: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report_text: str
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


def evaluate_classification_model(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3)

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    return ModelEvaluationResult(
        name=name,
        accuracy=acc,
        roc_auc=roc_auc,
        confusion_matrix=cm,
        classification_report_text=report,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )
