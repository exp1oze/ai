from sklearn.ensemble import RandomForestClassifier
from src.config import RANDOM_STATE


def build_random_forest(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        oob_score=True,
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return model


def train_random_forest(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def get_oob_score(model):
    return float(model.oob_score_)


def predict_random_forest(model, X_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba
