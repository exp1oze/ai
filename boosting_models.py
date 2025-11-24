from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from src.config import RANDOM_STATE

def build_adaboost(n_estimators=200, learning_rate=0.5):
    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
    )
    return model


def build_gradient_boosting(n_estimators=200, learning_rate=0.05, max_depth=3):
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
    )
    return model


def train_boosting_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict_boosting_model(model, X_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba
