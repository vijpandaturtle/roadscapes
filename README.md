# Roadscapes Dataset

A multitask multimodal dataset for autonomous driving scene understanding in diverse Indian road environments.

<p align="center">
  <img src="assets/day_grid.png" alt="Daytime Road Scenes" width="49%"/>
  <img src="assets/night_grid.png" alt="Nighttime Road Scenes" width="49%"/>
  <br/>
  <em>Sample scenes from the Roadscapes dataset: Daytime (left) and Nighttime (right)</em>
</p>

## Overview

Roadscapes is a comprehensive dataset designed to advance research in visual scene understanding for autonomous driving systems, with a particular focus on unstructured and diverse driving environments typical of Indian roads. The dataset consists of up to 9,000 annotated images capturing various road scenarios, accompanied by question-answer pairs for multiple vision-language tasks.

## Key Features

- **9,000+ images** from diverse Indian driving environments
- **Manually verified bounding boxes** for object detection
- **Scene attributes** derived through rule-based heuristics
- **Question-answer pairs** for multiple tasks:
  - Object grounding
  - Visual reasoning
  - Scene understanding
- **Diverse environments**:
  - Urban and rural settings
  - Highways and service roads
  - Village paths
  - Congested city streets
- **Varied conditions**: Daytime and nighttime captures

## Dataset Composition

The dataset has been specifically curated to represent the complexity and diversity of Indian road scenes, which often feature:
- Unstructured traffic patterns
- Mixed vehicle types (cars, motorcycles, auto-rickshaws, bicycles, etc.)
- Pedestrians and animals
- Varied road infrastructure
- Challenging lighting conditions

## Tasks Supported

1. **Object Grounding**: Locating and identifying objects in road scenes
2. **Visual Reasoning**: Understanding relationships and interactions between scene elements
3. **Scene Understanding**: Comprehensive interpretation of driving environments

## Getting Started

### Installation

```bash
git clone https://github.com/roadscapes/roadscapes_data.git
cd roadscapes_data
```

### Requirements

```bash
# Add your specific requirements here
pip install -r requirements.txt
```

### Dataset Structure

```
roadscapes_data/
├── image_data/
│   ├── images/
│   │   ├── test/
│   │   │   ├── Sequence_Day_n/
│   │   │   └── Sequence_Night_n/
│   │   └── train/
│   │       ├── Sequence_Day_n/
│   │       └── Sequence_Night_n/
├── annotations/
│   ├── vqa_train.csv
│   └── vqa_test.csv
└── README.md
```

## Usage

1. Install requirements on Python 3.10+ for the QA generation pipeline
   ```
   pip install tqdm
   ```
2. Run the generate_image_vqa.py for the generation pipeline
4. In order to run experiments use the notebook "Experiments.ipynb".
5. Data has been anonymized using the supporting code in directory "Data Preparation and Anonymization"
   

### Baseline Models

We provide baseline implementations using vision-language models for the image QA tasks. See the `baselines/` directory for:
- Model architectures
- Training scripts
- Evaluation metrics
- Pre-trained weights (if applicable)

## Data Collection and Annotation

### Collection Process
Images were captured across various Indian locations to ensure geographic and environmental diversity. The collection encompasses:
- Different times of day (daytime and nighttime)
- Various weather conditions
- Multiple road types and traffic densities

### Annotation Process
- **Bounding boxes**: Manually annotated and verified
- **Scene attributes**: Derived using rule-based heuristics
- **QA pairs**: Generated systematically based on scene attributes and objects

## Statistics

### Dataset Split

| Split | Daytime Images | Nighttime Images | Total |
|-------|----------------|------------------|-------|
| Train | 5,519 | 1,989 | 7,508 |
| Test | 1,277 | 196 | 1,473 |
| **Total** | **6,796** | **2,185** | **8,981** |

### Key Statistics
- **Total images**: 8,981
- **Daytime images**: 6,796 (75.7%)
- **Nighttime images**: 2,185 (24.3%)
- **Train/Test split**: Approximately 84% / 16%

## Benchmarks

Initial baseline results using vision-language models are provided in the paper. We encourage the community to improve upon these benchmarks.

### Model Performance (Accuracy)

| Model | Object Counting | Object Description | Surrounding Description |
|-------|----------------|-------------------|------------------------|
| GPT-4o | 0.598 | 0.495 | 0.701 |
| Paligemma | 0.187 | 0.501 | 0.485 |
| Phi-3.5 | **0.667** | 0.437 | 0.643 |
| GPT-4o-mini | 0.628 | 0.453 | 0.645 |

**Best performance by task:**
- **Object Counting**: Phi-3.5 (0.667)
- **Object Description**: Paligemma (0.501)
- **Surrounding Description**: GPT-4o (0.701)

## Citation

If you use Roadscapes in your research, please cite our paper:

```bibtex
@article{iyer2026roadscapesqa,
  title={RoadscapesQA: A Multitask, Multimodal Dataset for Visual Question Answering on Indian Roads},
  author={Iyer, Vijayasri and Rathinagiriswaran, Maahin and S, Jyothikamalesh},
  journal={arXiv preprint arXiv:2602.12877},
  year={2026}
}
```

## License

This repo is released under an MIT License. Whereas the dataset itself is released for non-commercial purposes only at the moment. 

## Contact

For questions, issues, or collaboration opportunities, please:
- Open an issue on this repository
- Contact: thisisvij98@gmail.com

## Acknowledgments

We thank all contributors involved in data collection, annotation, and verification. This work aims to advance autonomous driving research, particularly for challenging and unstructured environments.

<!-- ## Contributing

We welcome contributions to improve the dataset and baselines. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. -->

## Changelog

### Version 1.0 (Initial Release)
- 9,000 annotated images
- Bounding box annotations
- QA pairs for three task categories
- Baseline model implementations
