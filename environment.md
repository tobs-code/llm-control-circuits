# Environment Notes

This repository is a practical local research workspace rather than a packaged Python library, so setup is intentionally lightweight and script-oriented.

## Core Runtime

- Python `3.11`
- CUDA-capable PyTorch on Windows
- Typical development GPU: `RTX 2080 8GB`

## Main Dependencies

- `torch`
- `transformers`
- `transformer_lens`
- `numpy==1.26.4`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `circuitsvis`
- `accelerate`
- `sentencepiece`
- `protobuf`

## Practical Notes

- Several scripts use `trust_remote_code=True` for Hugging Face model families that require custom code.
- Local HF module cache is redirected into `.local/scratch/hf_modules`.
- Local-only generated vectors and caches live under `.local/`.
- Experiment outputs are generally written to `results/`.

## Suggested Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers transformer_lens numpy==1.26.4 scikit-learn matplotlib seaborn circuitsvis accelerate sentencepiece protobuf
```

If you use CPU-only or another CUDA version, adjust the PyTorch install command.

## Repo Layout

- `scripts/runs/`: experiment runners
- `scripts/analysis/`: post-hoc analysis scripts
- `scripts/demos/`: demos and visual examples
- `scripts/inspect/`: inspection helpers
- `scripts/utils/`: utility scripts
- `docs/`: long-form reports and notes
- `results/`: stored outputs
- `assets/`: figures and dashboards
- `.local/`: ignored local cache and side data
