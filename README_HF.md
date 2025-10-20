---
title: Hepatitis C Predictor
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.50.0"
app_file: app.py
pinned: false
license: mit
---

# Hepatitis C Predictor

An interactive machine learning application for Hepatitis C classification using PyTorch neural networks.

## Features

- 📊 **Data Exploration**: Interactive visualizations and statistics
- 🚀 **Model Training**: Train models with custom hyperparameters
- 📈 **Model Evaluation**: Comprehensive performance metrics
- 🤖 **Deep Learning**: PyTorch neural network with residual connections

## How to Use

1. **Data Exploration**: Browse the dataset, view distributions, and understand the features
2. **Model Training**: Adjust hyperparameters and train your own model
3. **Model Evaluation**: Evaluate performance with metrics, confusion matrix, and ROC curves

## Model Architecture

- **Input**: 12 clinical features (lab values + demographics)
- **Architecture**: Deep Neural Network with Residual Blocks
  - Hidden Layers: [128, 64, 32]
  - Residual Blocks: 2 per layer
  - Regularization: Layer Normalization + Dropout
- **Output**: Binary classification (Healthy vs. Hepatitis C)
- **Expected Accuracy**: ~97.5%

## Dataset

The dataset contains laboratory values from blood donors and Hepatitis C patients from the UCI Machine Learning Repository. It includes:

- **615 samples**
- **12 laboratory measurements**: ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT
- **Demographics**: Age, Sex
- **Target**: Binary classification (0=Healthy, 1=Hepatitis C)

The dataset is automatically downloaded on first run.

## Technical Details

- **Framework**: PyTorch 2.8+
- **UI**: Streamlit
- **ML Libraries**: scikit-learn, pandas, numpy
- **Visualization**: plotly, matplotlib, seaborn

## Source Code

Full source code and documentation available at: [GitHub Repository](https://github.com/Ninjalice/HEPATITIS_C_MODEL)

## Disclaimer

⚠️ **This model is for educational purposes only.** Do not use for actual medical diagnosis. Always consult healthcare professionals for medical advice.

## License

MIT License - See [LICENSE](https://github.com/Ninjalice/HEPATITIS_C_MODEL/blob/main/LICENSE) for details.
