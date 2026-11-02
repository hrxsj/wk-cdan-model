# CWRU WK-CDAN strict final-epoch validation

- Started: 2026-07-21T20:26:49+08:00
- Purpose: independent validation only; this directory does not overwrite any historical result.
- Script: `experiments/cwru_ieee_revision.py`
- Script SHA-256 at launch: `4C60F9BB4DEC6C94230D1791AE7C4C84AD169888BDB7881E4294281B8409B29A`
- Data protocol: `strict_holdout` (split each original continuous record 80/20 before windowing).
- Checkpoint policy: fixed final epoch 100; no target-test-based early stopping or model selection.
- Target-test policy: evaluated once after training for each task/seed.
- Batch size: 32.
- Seeds: 42, 43, 44, 45, 46.
- Tasks: all 12 directed cross-load pairs among 0, 1, 2, and 3 HP.
- Expected fits: 60.

Launch command:

```powershell
E:\JetBrains\Anaconda\envs\DL\python.exe -u experiments\cwru_ieee_revision.py --data-protocol strict_holdout --epochs 100 --batch-size 32 --methods "WK-CDAN Full" --tasks "1797:1772,1797:1750,1797:1730,1772:1797,1772:1750,1772:1730,1750:1797,1750:1772,1750:1730,1730:1797,1730:1772,1730:1750" --seeds "42,43,44,45,46" --skip-noise-ablation --device cuda --output-dir "runs\cwru_wkcdan_strict_final100_cross_20260721" --resume
```

The `--resume` flag skips only already completed task/seed rows in this dedicated directory.
