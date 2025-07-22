import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
import datetime

def normalize_label(label):
    if isinstance(label, str):
        return label.strip().lower()
    return label

def evaluate(csv_path, label_col="Emotion", prediction_col="prediction", plot_confusion=True, save_csv=True, output_path=None):
    df = pd.read_csv(csv_path)

    if label_col not in df.columns or prediction_col not in df.columns:
        raise ValueError(f"Missing columns: {label_col}, {prediction_col}")

    y_true = df[label_col].apply(normalize_label)
    y_pred = df[prediction_col].apply(normalize_label)

    print("\n📊 Evaluation Report:\n")
    report = classification_report(y_true, y_pred, digits=3, output_dict=True)
    acc = accuracy_score(y_true, y_pred)
    print(classification_report(y_true, y_pred, digits=3))
    print(f"✅ Accuracy: {acc:.3f}")

    # Save evaluation summary
    if save_csv:
        result_row = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": os.path.basename(csv_path),
            "accuracy": round(acc, 3),
            "macro_precision": round(report["macro avg"]["precision"], 3),
            "macro_recall": round(report["macro avg"]["recall"], 3),
            "macro_f1": round(report["macro avg"]["f1-score"], 3),
        }

        summary_path = Path(output_path or "results/eval_summary.csv")
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            summary_df = pd.concat([summary_df, pd.DataFrame([result_row])], ignore_index=True)
        else:
            summary_df = pd.DataFrame([result_row])

        summary_df.to_csv(summary_path, index=False)
        print(f"📁 Results saved to {summary_path}")

    # Plot confusion matrix
    if plot_confusion:
        labels = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to prediction CSV file")
    parser.add_argument("--label_col", default="Emotion", help="Ground truth column")
    parser.add_argument("--prediction_col", default="prediction", help="Prediction column")
    parser.add_argument("--output_path", default=None, help="Where to save the summary CSV")
    parser.add_argument("--no_plot", action="store_true", help="Disable confusion matrix plot")
    args = parser.parse_args()

    evaluate(
        csv_path=args.csv,
        label_col=args.label_col,
        prediction_col=args.prediction_col,
        plot_confusion=not args.no_plot,
        output_path=args.output_path,
    )
