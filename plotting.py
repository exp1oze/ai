from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.config import FIGURES_DIR

def ensure_figures_dir_exists():
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

def plot_confusion_matrix(cm, class_names, title, filename):
    ensure_figures_dir_exists()

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="настоящий класс",
        xlabel="предсказанный класс",
        title=title,
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path = Path(FIGURES_DIR) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(results, filename="roc_curves.png"):
    ensure_figures_dir_exists()

    fig, ax = plt.subplots(figsize=(6, 5))

    for res in results:
        ax.plot(res.fpr, res.tpr, label=f"{res.name} (AUC = {res.roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="случайное угадывание (AUC = 0.5)")

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC")
    ax.legend(loc="lower right")
    ax.grid(True)

    out_path = Path(FIGURES_DIR) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
