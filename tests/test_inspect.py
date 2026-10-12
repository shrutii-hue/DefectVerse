import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.patchcore import PatchCore

model = PatchCore.load("outputs/models/patchcore_bottle.pt")
print("Saved model threshold:", getattr(model, "threshold", None))
print("Saved min_score:", getattr(model, "min_score", None))
print("Saved max_score:", getattr(model, "max_score", None))

print("\n--- Training Images (data/bottle/train/good) ---")
train_scores = []
for p in sorted(Path("data/bottle/train/good").glob("*.png")):
    res = model.predict(p)
    train_scores.append(res["anomaly_score"])
    print(f"  {p.name}: score={res['anomaly_score']:.4f}, is_anomaly={res['is_anomaly']}, max={res['anomaly_map'].max():.4f}")

print("\n--- Test Good Images (data/bottle/test/good) ---")
test_good_scores = []
for p in sorted(Path("data/bottle/test/good").glob("*.png")):
    res = model.predict(p)
    test_good_scores.append(res["anomaly_score"])
    print(f"  {p.name}: score={res['anomaly_score']:.4f}, is_anomaly={res['is_anomaly']}")

print("\n--- Test Defect Images (broken_large) ---")
test_defect_scores = []
for p in sorted(Path("data/bottle/test/broken_large").glob("*.png")):
    res = model.predict(p)
    test_defect_scores.append(res["anomaly_score"])
    print(f"  {p.name}: score={res['anomaly_score']:.4f}, is_anomaly={res['is_anomaly']}")

print(f"\nSummary:")
print(f"  Train Good Scores: min={min(train_scores):.4f}, max={max(train_scores):.4f}, mean={sum(train_scores)/len(train_scores):.4f}")
print(f"  Test Good Scores: min={min(test_good_scores):.4f}, max={max(test_good_scores):.4f}, mean={sum(test_good_scores)/len(test_good_scores):.4f}")
print(f"  Test Defect Scores: min={min(test_defect_scores):.4f}, max={max(test_defect_scores):.4f}, mean={sum(test_defect_scores)/len(test_defect_scores):.4f}")
