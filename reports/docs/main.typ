#import "@preview/elsearticle:1.0.0": *


#show: elsearticle.with(
  title: "Hepatitis C Prediction Model: Data Analysis and Neural Network Implementation - Practice 2",
  authors: (
    (
      name: "Igor Vons",
    ),
    (
      name: "Wassim Bouzarhoun",
    ),
    (
      name: "Endika Aguirre",
    ),
  ),
  
  format: "preprint", // Change from "review" to "preprint" or "final"
)


= Dataset Overview

The hepatitis C dataset contains 615 samples with the following features:
- Laboratory measurements: ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT
- Demographic information: Age, Sex
- Target variable: Disease category (Healthy vs Hepatitis C)
- An index column (Patient ID)

= Data Exploration

== Data Quality Assessment

The dataset quality analysis reveals several important characteristics:

=== Missing Values Analysis
The dataset contains missing values in several laboratory measurements, as shown in @tab:missing-values.

#figure(
  table(
    columns: 2,
    table.header([*Feature*], [*Missing Count*]),
    [ALB], [1],
    [ALP], [18],
    [ALT], [1],
    [CHOL], [10],
    [PROT], [1],
  ),
  caption: [Missing values by feature]
) <tab:missing-values>

=== Target Distribution
The distribution of disease categories shows a significant class imbalance, as detailed in 

The analysis shows that the dataset is heavily imbalanced, with healthy blood donors comprising 87.8% of the samples, while the three disease categories make up only 12.2% of the data. This imbalance must be addressed during model training.

== Feature Distributions
The violin plots in @fig:violin-plots show the distribution of each laboratory measurement.

#figure(
  image("../figures/violin_plots.png", width: 100%),
  caption: [Distribution of laboratory measurements with outliers highlighted]
) <fig:violin-plots>

== Correlation Analysis
Reviewing the correlation matrix, we don't find much strong dependeces. The most notable correlations are: PROT and ALB (0.56 positive), GGT and AST (0.49 positive), GGT and ALP (0.45 positive), CHOL and CHE (0.43 positive), Age and Patient ID (0.42 positive)

= Data Preprocessing

The data preprocessing steps included:
1. Handling missing values using median imputation.
2. Encoding the categorical variable "Sex" using one-hot encoding.
3. Scaling numerical features using StandardScaler.
4. Splitting the data into training and testing sets (80/20 split).

= Model Development

The neural network model was developed using PyTorch and trained on the preprocessed dataset. The architecture shape is (12-128-64-32-2) consists of an input layer, two hidden layers with ReLU activation, and an output layer with a sigmoid activation function for binary classification.

= Model Evaluation

== Prediction Performance
Currently, the model achieves a great accuracy for the No Hepatitis C class, but struggles with the Hepatitis C class due to the significant class imbalance in the dataset. The confusion matrix in @fig:confusion-matrix illustrates this performance.

#figure(
  image("../figures/confusion_matrix.png", width: 60%),
  caption: [Confusion Matrix of the Neural Network Model]
) <fig:confusion-matrix>

== Calibration Analysis
The calibration curve in @fig:calibration-curve indicates that the model's predicted probabilities are not well-calibrated, particularly around 0.5. This suggests that the model tends to be overconfident in its predictions of No Hepatitis C cases.

#figure(
  image("../figures/calibration_curve.png", width: 50%),
  caption: [Calibration Curve of the Neural Network Model]
) <fig:calibration-curve>
