# Hepatitis C Predictor

A machine learning project to predict Hepatitis C using PyTorch neural networks.

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://ninjalice.github.io/HEPATITIS_C_MODEL/src.html)

## Project Organization

    ├── data/
    │   ├── raw/              <- The original, immutable data dump
    │   └── processed/        <- The final, canonical data sets for modeling
    │
    ├── models/               <- Trained and serialized models
    │
    ├── notebooks/            <- Jupyter notebooks for analysis and modeling
    │   ├── 01-data-exploration.ipynb
    │   ├── 02-data-preprocessing.ipynb
    │   ├── 03-model-training.ipynb
    │   └── 04-model-prediction.ipynb
    │
    ├── reports/              <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures/          <- Generated graphics and figures
    │
    ├── src/                  <- Source code for use in this project
    │   ├── __init__.py
    │   ├── data.py           <- Scripts to download or generate data
    │   ├── features.py       <- Scripts to turn raw data into features
    │   ├── models.py         <- Scripts to train models and make predictions
    │   └── visualization.py  <- Scripts to create exploratory visualizations
    │
    ├── requirements.txt      <- The requirements file for reproducing the environment
    └── README.md            <- The top-level README for developers


## Docs

You can check the modules docs in the docs folder or directly from the deployed version on GH pages here: https://ninjalice.github.io/HEPATITIS_C_MODEL/src.html

## Getting Started

1. Install dependencies:

   ```bash
   uv sync --frozen
   ```
   Alternatively ```bash pip install -r requirements.txt``` should also work.

2. Download the dataset from Kaggle and place it in `data/raw/hepatitis_data.csv`

3. Follow the notebooks in order:
   - `01-data-exploration.ipynb` - Explore the dataset
   - `02-data-preprocessing.ipynb` - Clean and prepare data
   - `03-model-training.ipynb` - Train the neural network
   - `04-model-prediction.ipynb` - Make predictions on new data(WIP)

## Dataset

The dataset contains laboratory values from blood donors and Hepatitis C patients:

- **Source**: https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset
- **Size**: 615 samples
- **Features**: 12 laboratory measurements + age and sex
- **Target**: Binary classification (Healthy vs Hepatitis C)

## Model

- **Architecture**: Deep Neural Network (12 → 128 → 64 → 32 → 2)
- **Framework**: PyTorch
- **Expected Accuracy**: ~97.5%

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Run tests and ensure documentation is updated
5. Commit your changes:
   ```bash
   git commit -m "Add detailed description of your changes"
   ```
6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Create a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Include docstrings for all functions and classes
- Add comments for complex logic
- Update documentation when changing functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Important Note

⚠️ This model is for educational purposes only. Do not use for actual medical diagnosis. Always consult healthcare professionals.
