"""Evaluation metrics and benchmark utilities for DefectVerse."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def evaluate_dataset(
    model: Any,
    test_dir: Path,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run inference across test dataset and compute performance metrics.

    Args:
        model: Fitted PatchCore model instance.
        test_dir: Path to test dataset directory containing category subfolders.
        threshold: Decision threshold (uses model.threshold if None).

    Returns:
        Dictionary of evaluation results, metrics, and per-sample predictions.
    """
    test_dir = Path(test_dir).resolve()
    if threshold is None:
        threshold = model.threshold

    y_true: List[int] = []
    y_scores: List[float] = []
    y_pred: List[int] = []
    sample_details: List[Dict[str, Any]] = []

    # Map categories to labels (0 = normal/good, 1 = anomaly)
    subdirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    
    start_eval_time = time.time()
    for cat_dir in subdirs:
        cat_name = cat_dir.name
        is_defect_cat = (cat_name.lower() != "good")
        label = 1 if is_defect_cat else 0

        image_files = sorted(
            list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.jpeg"))
        )

        for img_path in image_files:
            t0 = time.time()
            pred_dict = model.predict(img_path)
            inference_ms = (time.time() - t0) * 1000.0

            score = float(pred_dict["anomaly_score"])
            predicted_label = 1 if score >= threshold else 0

            y_true.append(label)
            y_scores.append(score)
            y_pred.append(predicted_label)

            sample_details.append(
                {
                    "path": str(img_path),
                    "filename": img_path.name,
                    "category": cat_name,
                    "true_label": label,
                    "predicted_label": predicted_label,
                    "score": score,
                    "is_correct": (label == predicted_label),
                    "inference_ms": inference_ms,
                }
            )

    total_eval_time = time.time() - start_eval_time

    # Calculate metrics
    y_true_np = np.array(y_true)
    y_scores_np = np.array(y_scores)
    y_pred_np = np.array(y_pred)

    # ROC & AUC
    if len(np.unique(y_true_np)) > 1:
        fpr, tpr, roc_thresholds = roc_curve(y_true_np, y_scores_np)
        roc_auc = float(auc(fpr, tpr))
    else:
        fpr, tpr, roc_thresholds = np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([0.0, 1.0])
        roc_auc = 1.0

    # Classification Metrics
    precision = float(precision_score(y_true_np, y_pred_np, zero_division=0))
    recall = float(recall_score(y_true_np, y_pred_np, zero_division=0))
    f1 = float(f1_score(y_true_np, y_pred_np, zero_division=0))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = float((tp + tn) / max(1, len(y_true_np)))
    specificity = float(tn / max(1, tn + fp))

    # Category-level statistics
    category_stats = {}
    for cat_dir in subdirs:
        cat_name = cat_dir.name
        cat_samples = [s for s in sample_details if s["category"] == cat_name]
        cat_count = len(cat_samples)
        cat_correct = sum(1 for s in cat_samples if s["is_correct"])
        category_stats[cat_name] = {
            "total": cat_count,
            "correct": cat_correct,
            "accuracy": (cat_correct / max(1, cat_count)),
            "mean_score": float(np.mean([s["score"] for s in cat_samples])) if cat_samples else 0.0,
        }

    return {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_thresholds": roc_thresholds.tolist(),
        "confusion_matrix": cm.tolist(),
        "counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "category_stats": category_stats,
        "sample_details": sample_details,
        "total_samples": len(y_true),
        "total_time_seconds": total_eval_time,
        "mean_latency_ms": float(np.mean([s["inference_ms"] for s in sample_details])),
        "applied_threshold": threshold,
    }
