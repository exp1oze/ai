from pathlib import Path

import src.preprocessing as preprocessing
from src.boosting_models import (
    build_adaboost,
    build_gradient_boosting,
    predict_boosting_model,
    train_boosting_model,
)
from src.config import FIGURES_DIR
from src.metrics import evaluate_classification_model
from src.plotting import plot_confusion_matrix, plot_roc_curves
from src.random_forest_model import (
    build_random_forest,
    get_oob_score,
    predict_random_forest,
    train_random_forest,
)



def main():
    df = preprocessing.load_data()
    print(f"исходный датасет: {df.shape[0]} объектов, {df.shape[1]} признаков")

    X_train, X_test, y_train, y_test = preprocessing.train_test_split_data(df)
    print(f"размер train: {X_train.shape[0]}, размер test: {X_test.shape[0]}")

    print("\nRANDOM_FOREST")
    rf_model = build_random_forest()
    rf_model = train_random_forest(rf_model, X_train, y_train)

    oob_score = get_oob_score(rf_model)
    print(f"RANDOM FOREST OOB ACCURACY: {oob_score:.3f}")

    rf_y_pred, rf_y_proba = predict_random_forest(rf_model, X_test)
    rf_eval = evaluate_classification_model(
        name="RANDOM FOREST",
        y_true=y_test,
        y_pred=rf_y_pred,
        y_proba=rf_y_proba,
    )

    print(f"тестовая accuracy (Random Forest): {rf_eval.accuracy:.3f}")
    print(rf_eval.classification_report_text)

    print("\nADABOOST")
    ada_model = build_adaboost()
    ada_model = train_boosting_model(ada_model, X_train, y_train)

    ada_y_pred, ada_y_proba = predict_boosting_model(ada_model, X_test)
    ada_eval = evaluate_classification_model(
        name="ADABOOST",
        y_true=y_test,
        y_pred=ada_y_pred,
        y_proba=ada_y_proba,
    )

    print(f"тестовая accuracy (AdaBoost): {ada_eval.accuracy:.3f}")
    print(ada_eval.classification_report_text)

    print("\nGRADIENT_BOOSTING")
    gb_model = build_gradient_boosting()
    gb_model = train_boosting_model(gb_model, X_train, y_train)

    gb_y_pred, gb_y_proba = predict_boosting_model(gb_model, X_test)
    gb_eval = evaluate_classification_model(
        name="Gradient Boosting",
        y_true=y_test,
        y_pred=gb_y_pred,
        y_proba=gb_y_proba,
    )

    print(f"тестовая accuracy (Gradient Boosting): {gb_eval.accuracy:.3f}")
    print(gb_eval.classification_report_text)

    print("\nconfusion matrices + ROC")
    class_names = ["нет болезни", "болезнь"]

    figures_dir = Path(FIGURES_DIR)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        rf_eval.confusion_matrix,
        class_names=class_names,
        title="Random Forest",
        filename="confusion_matrix_random_forest.png",
    )
    plot_confusion_matrix(
        ada_eval.confusion_matrix,
        class_names=class_names,
        title="AdaBoost",
        filename="confusion_matrix_adaboost.png",
    )
    plot_confusion_matrix(
        gb_eval.confusion_matrix,
        class_names=class_names,
        title="Gradient Boosting",
        filename="confusion_matrix_gradient_boosting.png",
    )

    all_results = [rf_eval, ada_eval, gb_eval]
    plot_roc_curves(all_results, filename="roc_curves.png")

    print(f"loaded in: {figures_dir.resolve()}")

if __name__ == "__main__":
    main()
