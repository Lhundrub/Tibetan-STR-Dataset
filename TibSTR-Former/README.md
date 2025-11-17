# TibNST: Tibetan Natural Scene Text Recognition

TibNST is a complete research and engineering effort for Tibetan natural scene text recognition (STR). It covers data preparation, training, evaluation, ablation studies, and a detection branch. The project is built on PyTorch/Lightning, supports multiple CNN + Transformer/BiLSTM architectures, and ships with commonly used analysis and visualization scripts.

## Highlights
- **Large-scale dataset**: Includes scripts for splitting TibNST into training/validation/testing sets, using `data/train_new.json` and `data/val_new.json` by default.
- **Multiple backbones**: `model_v0`–`model_v4` span CNN+Transformer, residual, FPN, attention, and BiLSTM variants.
- **Lightning Fabric training pipeline**: `train_complete.py` encapsulates augmentations, metrics, and logging.
- **Complete evaluation toolkit**: `test.py` handles test-set splitting and inference; `ablation_study.py` aggregates multi-model performance; `metrics_paper.py` and `test_char_coverage.py` provide deeper analysis.
- **Detection extension**: `detect.py` plus `detect_evaluate.py` reproduce the Faster R-CNN pipeline, enabling full text detection + recognition experiments.

---

## TibNST Dataset

TibNST is the first large-scale annotated dataset dedicated to Tibetan natural scene text recognition, created to fill the gap in standardized benchmarks and sufficient training data for Tibetan STR research. Key characteristics:

- **Scale and sources**: 2,049 real-world images collected from street signs, temple plaques, commercial boards, handwritten walls, and other media, covering diverse lighting, viewpoints, backgrounds, blur, and occlusion.
- **Annotation quality**: Native Tibetan speakers proficient in traditional and modern scripts transcribed each instance in LabelStudio, storing all text as Unicode sequences.
- **Character statistics**: 45,473 total characters; each image contains 1–92 characters, covering stacked, cursive, italic, stretched, and other complex layouts.
- **Task difficulty**: Samples vary dramatically in font, weight, size, layout, and background complexity, faithfully reproducing real-world noise and interference. Detailed statistics are provided in Appendix B of the paper.
- **Download**: Zenodo — https://zenodo.org/records/17599952


## Repository Structure
.
├── data/                     # Main data (train_new.json / val_new.json)
├── new-data/                 # Raw or incremental data (can be merged via scripts)
├── images/, paper_figures/   # Debug and paper figures
├── models/                   # Model definitions (model_v0.py ... model_v4.py)
├── train_complete.py         # Primary training entry point
├── test.py                   # Validation/test split and evaluation
├── ablation_study.py         # Multi-model ablation + visualization
├── quick_backbone_attention_ablation.py
├── detect.py / detect_evaluate.py / dataset.py   # Detection branch
├── data_loader.py / ocr_transforms.py            # Dataset and augmentations
├── debug_test/               # Diagnostics for character coverage, augmentations, etc.
├── requirements.txt
└── README.md



Acknowledgements
If this repository or the TibNST dataset helps your research, please cite it and let us know via an issue.