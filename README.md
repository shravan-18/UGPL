# UGPL: Uncertainty-Guided Progressive Learning for Evidence-Based Classification in Computed Tomography

<div align="center">
<img src="assets/ICCV_2025_logo.png" width="75%"/>
</div>

> **UGPL: Uncertainty-Guided Progressive Learning for Evidence-Based Classification in Computed Tomography**  
> *Shravan Venkatraman\*, Pavan Kumar S\*, Rakesh Raj\*, Chandrakala S*  
> Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops 2025  
> *(\* Equal Contribution)*

#### [project page](https://github.com/shravan-18/UGPL) | [paper](https://github.com/shravan-18/UGPL)

## Abstract

Accurate classification of computed tomography (CT) images is essential for diagnosis and treatment planning, but existing methods often struggle with the subtle and spatially diverse nature of pathological features. Current approaches typically process images uniformly, limiting their ability to detect localized abnormalities that require focused analysis. We introduce UGPL, an uncertainty-guided progressive learning framework that performs a global-to-local analysis by first identifying regions of diagnostic ambiguity and then conducting detailed examination of these critical areas. Our approach employs evidential deep learning to quantify predictive uncertainty, guiding the extraction of informative patches through a non-maximum suppression mechanism that maintains spatial diversity. This progressive refinement strategy, combined with an adaptive fusion mechanism, enables UGPL to integrate both contextual information and fine-grained details. Experiments across three CT datasets demonstrate that UGPL consistently outperforms state-of-the-art methods, achieving improvements of 3.29%, 2.46%, and 8.08% in accuracy for kidney abnormality, lung cancer, and COVID-19 detection, respectively. Our analysis shows that the uncertainty-guided component provides substantial benefits, with performance dramatically increasing when the full progressive learning pipeline is implemented.

<div align="center">
  <img src="assets/architecture.png" width="90%" alt="UGPL Architecture">
  <p>Figure: Overview of the UGPL architecture pipeline</p>
</div>

## Key Features

- **Uncertainty-Guided Analysis**: Dynamically allocates computational resources to regions of high diagnostic ambiguity using evidential deep learning
- **Adaptive Patch Extraction**: Selects diverse, non-overlapping regions for detailed analysis through a non-maximum suppression mechanism
- **Progressive Refinement Strategy**: Combines global contextual information with localized fine-grained details
- **Multi-Component Loss Formulation**: Jointly optimizes classification accuracy, uncertainty calibration, and spatial diversity

## Installation

```bash
# Clone the repository
git clone https://github.com/username/UGPL.git
cd UGPL

# Create a conda environment (optional)
conda create -n ugpl python=3.8
conda activate ugpl

# Install dependencies
pip install -r requirements.txt

# Set execution permissions for scripts
chmod +x scripts/*.sh
```

## Dataset Preparation

The model is trained and evaluated on three CT image datasets:
- Kidney Abnormality Dataset
- Lung Cancer CT Dataset
- COVID-19 CT Dataset

Place your datasets in the following structure:

```
data/
├── kidney/
│   ├── Normal/
│   ├── Cyst/
│   ├── Tumor/
│   └── Stone/
├── lung/
│   ├── benign/
│   ├── malignant/
│   └── normal/
└── covid/
    ├── covid/
    └── nonCovid/
```

## Usage

### Training

To train the UGPL model from scratch:

```bash
./scripts/train.sh --dataset kidney --kidney_path data/kidney/
```

You can customize training parameters:

```bash
./scripts/train.sh --dataset lung \
                   --lung_path data/lung/ \
                   --batch_size 16 \
                   --epochs 100 \
                   --lr 0.0001 \
                   --backbone resnet34 \
                   --num_patches 3 \
                   --use_ema \
                   --early_stopping
```

### Evaluation

To evaluate a trained model:

```bash
./scripts/evaluate.sh --dataset covid \
                      --covid_path data/covid/ \
                      --checkpoint checkpoints/covid_best_model.pth \
                      --analyze_errors
```

### Visualization

Generate visualizations for a trained model:

```bash
./scripts/visualize.sh --dataset kidney \
                       --checkpoint checkpoints/kidney_best_model.pth
```



### Hyperparameter Tuning

Run hyperparameter tuning:

```bash
./scripts/hyperparameter_tuning.sh --dataset lung \
                                   --hpt_method bayesian \
                                   --hpt_trials 20
```

## Model Components

- **Global Uncertainty Estimator**: Performs initial classification and generates pixel-wise uncertainty maps
- **Progressive Patch Extractor**: Selects high-uncertainty regions for detailed analysis
- **Local Refinement Network**: Conducts high-resolution analysis of extracted patches
- **Adaptive Fusion Module**: Integrates global and local predictions using learned weights

## Results

<div align="center">
  <img src="assets/components_viz.png" width="50%" alt="Results Comparison">
  <p>Figure: Visual comparison of UGPL components across datasets</p>
</div>

### Quantitative Results

| Model | Kidney Abnormalities |  | Lung Cancer Type |  | COVID Presence |  |
|-------|--------------------|----|---------------|----|---------------|----|
|       | Accuracy | F1 | Accuracy | F1 | Accuracy | F1 |
| Global Model | 0.9811 | 0.9746 | 0.9617 | 0.9611 | 0.7108 | 0.7078 |
| Local Model | 0.4057 | 0.1443 | 0.5122 | 0.2258 | 0.6486 | 0.6343 |
| Fused Model (UGPL) | **0.9971** | **0.9946** | **0.9817** | **0.9764** | **0.8108** | **0.7903** |

### Ablation Study

| Configuration | Kidney Abnormalities |  | Lung Cancer Type |  | COVID Presence |  |
|---------------|--------------------|----|---------------|----|---------------|----|
|               | Accuracy | F1 | Accuracy | F1 | Accuracy | F1 |
| Global-only | 0.5676 | 0.5545 | 0.5000 | 0.3890 | 0.2535 | 0.1495 |
| No UG | 0.5766 | 0.5558 | 0.4634 | 0.3764 | 0.2363 | 0.1536 |
| Fixed Patches | 0.5766 | 0.5697 | 0.4573 | 0.3731 | 0.2347 | 0.1533 |
| Full Model | **0.9971** | **0.9945** | **0.9817** | **0.9764** | **0.8108** | **0.7903** |

## Project Structure

```
UGPL/
├── scripts/              # Shell scripts for training, evaluation, etc.
├── src/                  # Source code
│   ├── data/             # Dataset and dataloader modules
│   ├── models/           # Model architecture definitions
│   ├── training/         # Training loops and loss functions
│   ├── evaluation/       # Evaluation metrics and visualizations
│   ├── utils/            # Utility functions
│   └── analysis/         # Performance analysis tools
├── main.py               # Main entry point
└── README.md             # This file
```

## Citation

If you find our work useful for your research, please consider citing:

```bibtex
@InProceedings{UGPL2025,
  author    = {Venkatraman, Shravan and Kumar S, Pavan and Raj, Rakesh and S, Chandrakala},
  title     = {UGPL: Uncertainty-Guided Progressive Learning for Evidence-Based Classification in Computed Tomography},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
  month     = {October},
  year      = {2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- We acknowledge the contributions of previous works in uncertainty quantification and medical image analysis
- We thank the authors of the Kidney, Lung, and COVID-19 CT datasets for making their data publicly available
