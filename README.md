# Locomotion-Aware

Source code for **"Locomotion–Aware Deep Gesture Recognition"**

## 📄 Publication Status

The corresponding paper is currently under review for publication.  
The source code will be publicly released after the paper is accepted and published to ensure integrity and compliance with publisher policies.

## 📦 Dataset

The radar dataset used in this study is publicly available on IEEE DataPort:

**DSMU-RAD: Dynamic, Stationary, and Walking Users FMCW Radar Dataset**  
📘 DOI: [10.21227/3hvh-bf32](https://dx.doi.org/10.21227/3hvh-bf32)

## Method overview

- **AIS preprocessing** (Algorithm 1): adaptive Doppler thresholding → Range–Angle Projection (RAP).
- **Modified MambaVision** backbone for **150-channel** radar tensors (`150 × 32 × 64`).
- **Multi-task learning**: gesture classification + locomotion auxiliary head.
- **Supervised contrastive** regularization (gesture-only positives, τ = 0.07).
- **Two-phase training**: λ = 1.0 then fine-tune with λ = 0.1.

## Repository layout

```text
Locomotion-Aware/
├── configs/default.yaml
├── lmadgr/
│   ├── models/          # MambaVision multi-task + ablation variants
│   ├── data/            # dataset loader + AIS
│   ├── losses/          # supervised contrastive
│   └── baselines/       # DI-Gesture / MLFF style stubs
├── scripts/
│   ├── preprocess_ais.py
│   ├── train.py                 # two-phase full model
│   ├── train_ablation.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## Installation

```bash
conda create -n lmadgr python=3.10 -y
conda activate lmadgr
pip install -r requirements.txt
```

The backbone is loaded from Hugging Face (see `configs/default.yaml` for `model_name`). First run will download model weights.

## Data (LAGID)

Organize preprocessed RAP `.npy` files as:

```text
LAGID/
├── gesture_walking/   # locomotion id 0
├── gesture_still/     # locomotion id 1
└── motion/            # locomotion id 2 (non-gesture body motion)
```

Filename convention: `prefix_id_Gesture Name.npy` (see `lmadgr/data/dataset.py`).

Edit `configs/default.yaml`:

```yaml
data:
  walking_dir: /path/to/DSMU-RAD/gesture_walking
  still_dir: /path/to/DSMU-RAD/gesture_still
  motion_dir: /path/to/DSMU-RAD/motion
```

### Optional: run AIS on Range–Doppler cubes

```bash
python scripts/preprocess_ais.py \
  --input-dir /path/to/rd_cubes \
  --output-dir /path/to/rap_npy \
  --config configs/default.yaml
```

## Training

### Full model (paper two-phase schedule)

```bash
python scripts/train.py --config configs/default.yaml --phase all
```

Or run phases separately:

```bash
python scripts/train.py --config configs/default.yaml --phase 1
python scripts/train.py --config configs/default.yaml --phase 2 \
  --resume checkpoints/phase1_best.pth
```

### Ablations

```bash
python scripts/train_ablation.py --variant baseline
python scripts/train_ablation.py --variant movement
python scripts/train_ablation.py --variant contrastive
python scripts/train_ablation.py --variant full
```

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best_model.pth \
  --out-dir results
```

## Model details

| Component | Spec |
|-----------|------|
| Backbone | MambaVision-L-21K (hidden 1568) |
| Input | 150 × 32 × 64 |
| Gesture head | Linear → 13 classes |
| Locomotion head | 47040 → 2048 → 3 |
| Contrastive head | 1568 → 2048 → 1024 |
| Loss | CE_gest + λ_mo CE_mo + λ_sc SC |

## 📬 Contact

For inquiries related to the dataset or source code, please contact:

**A. Kajbaf**  
Email: a.kajbaf@ec.iut.ac.ir  
Affiliation: University of Tehran & Isfahan University of Technology

## 📚 Citation

If you use this dataset or related materials, please cite:

> A. Kajbaf, “DSMU-RAD: Dynamic, Stationary, and Walking Users FMCW Radar Dataset,” IEEE DataPort, 2025.  
> DOI: [10.21227/3hvh-bf32](https://dx.doi.org/10.21227/3hvh-bf32)

```bibtex
@dataset{kajbaf2025dsmurad,
  author    = {Kajbaf, A.},
  title     = {{DSMU-RAD}: Dynamic, Stationary, and Walking Users {FMCW} Radar Dataset},
  year      = {2025},
  publisher = {IEEE DataPort},
  doi       = {10.21227/3hvh-bf32},
  url       = {https://dx.doi.org/10.21227/3hvh-bf32}
}
```

## License

Code released for research use accompanying the paper. Please contact the authors for other licensing requests.
