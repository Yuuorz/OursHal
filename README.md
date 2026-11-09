# OursHal

Research code for visual hallucination evaluation and diagnosis on COCO POPE.

- `run.py`, `train.py`, `restorer.py`, `visual_restore.py`: Qwen2-VL inference and restoration experiments.
- `evaluate.py`: answer parsing and evaluation metrics.
- `attention_bias.py`: attention intervention experiments.
- `llava_baseline.py`: Hugging Face LLaVA baseline.
- `llava_native_baseline.py`: original LLaVA baseline with configurable output directory and prompt suffix.
- `attention_gt.py`: layerwise attention localization against COCO bounding boxes.
- `test_restorer.py`: existing restoration regression tests.

Model weights, datasets, experiment outputs and local project notes are kept outside Git.
The scripts currently reference local model, dataset and original LLaVA source paths;
adjust those paths before running on another machine.

Qwen/HF and original LLaVA use separate environments. The completed experiments used
Transformers 4.57.6 / PyTorch 2.10.0 for Qwen/HF, and Transformers 4.31.0 /
PyTorch 2.0.1 for original LLaVA. `requirements.txt` contains the existing project
requirements, not a lockfile for both environments.
