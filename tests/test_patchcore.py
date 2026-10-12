import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from src.models.patchcore import PatchCore



def test_patchcore_lifecycle():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchCore(backbone_name="resnet18", coreset_sampling_ratio=0.5, device=device)

    data_dir = Path("data/bottle")
    train_dir = data_dir / "train" / "good"
    assert train_dir.exists(), "Train directory data/bottle/train/good does not exist."

    train_paths = [str(p) for p in train_dir.glob("*.png")]
    assert len(train_paths) > 0, "No training images found."

    # 1. Fit memory bank
    model.fit(train_paths)
    assert model.memory_bank is not None
    assert model.memory_bank.ndim == 2

    # 2. Predict on normal and defect samples
    test_normal = next((data_dir / "test" / "good").glob("*.png"))
    test_defect = next((data_dir / "test" / "broken_large").glob("*.png"))

    res_normal = model.predict(test_normal)
    res_defect = model.predict(test_defect)

    assert "anomaly_score" in res_normal
    assert "is_anomaly" in res_normal
    assert "heatmap_overlay" in res_normal
    assert isinstance(res_normal["anomaly_score"], float)
    assert res_normal["heatmap_overlay"].shape[:2] == res_normal["original_image"].shape[:2]

    # 3. Save and load
    save_path = Path("outputs/models/test_patchcore.pt")
    model.save(save_path)
    assert save_path.exists()

    loaded_model = PatchCore.load(save_path, device=device)
    loaded_res = loaded_model.predict(test_defect)

    assert abs(res_defect["anomaly_score"] - loaded_res["anomaly_score"]) < 1e-4

    # 4. Out-of-Distribution (OOD) Guardrail Test
    assert res_defect["is_ood"] is False, "In-distribution bottle should not trigger OOD guardrail."
    ood_noise = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    res_ood = model.predict(ood_noise)
    assert res_ood["is_ood"] is True, "Random image should trigger OOD guardrail."
    assert "[GUARDRAIL TRIGGERED]" in res_ood["guardrail_message"]
    print("OOD Guardrail verified: successfully intercepted out-of-distribution input!")

    # Cleanup test checkpoint
    if save_path.exists():
        save_path.unlink()


    print("All PatchCore lifecycle tests passed successfully!")


if __name__ == "__main__":
    test_patchcore_lifecycle()
