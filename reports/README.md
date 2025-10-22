# Reports and Visualizations

This directory contains generated analysis, visualizations, and reports from the Hepatitis C classification project.

## 📊 Generated Figures

### Data Analysis Figures

#### `data_overview.png`
- **Purpose**: Provides a comprehensive overview of the dataset structure
- **Contents**: Sample distributions, missing values analysis, basic statistics
- **Interpretation**: Helps understand data quality and preprocessing needs
- **Key Insights**: Shows balanced dataset with minimal missing values

#### `feature_distributions.png`
- **Purpose**: Visualizes the distribution of all laboratory measurements
- **Contents**: Histograms for each of the 12 laboratory features
- **Interpretation**: 
  - Normal vs skewed distributions
  - Outlier detection
  - Feature scaling requirements
- **Clinical Relevance**: Different lab values have different normal ranges and distributions

#### `correlation_matrix.png`
- **Purpose**: Shows relationships between different laboratory measurements
- **Contents**: Heatmap of correlation coefficients between all features
- **Interpretation**:
  - **Strong correlations** (>0.7): Indicate related lab measurements
  - **Weak correlations** (<0.3): Independent measurements
  - **Negative correlations**: Inverse relationships between lab values
- **Clinical Insights**: Some liver enzymes naturally correlate with each other

#### `violin_plots.png`
- **Purpose**: Compares feature distributions between healthy and Hepatitis C patients
- **Contents**: Violin plots showing distribution shapes for each class
- **Interpretation**:
  - **Overlapping distributions**: Features with less discriminative power
  - **Separated distributions**: Highly predictive features
  - **Distribution shapes**: Normal vs bimodal vs skewed patterns
- **Medical Significance**: Shows which lab values are most affected by Hepatitis C

### Model Performance Figures

#### `training_history.png`
- **Purpose**: Tracks model learning progress during training
- **Contents**: Training and validation loss/accuracy curves over epochs
- **Interpretation**:
  - **Convergence**: Both curves stabilizing indicates good training
  - **Overfitting**: Large gap between training and validation performance
  - **Underfitting**: Both curves plateauing at poor performance
- **Quality Indicators**: Smooth curves with minimal oscillation show stable learning

#### `confusion_matrix.png`
- **Purpose**: Detailed breakdown of model predictions vs actual labels
- **Contents**: 2x2 matrix showing True Positives, False Positives, etc.
- **Interpretation**:
  - **Diagonal values**: Correct predictions (higher is better)
  - **Off-diagonal values**: Prediction errors (lower is better)
  - **Sensitivity**: True Positive Rate (crucial for medical diagnosis)
  - **Specificity**: True Negative Rate (avoiding false alarms)
- **Clinical Impact**: False negatives are particularly concerning in medical diagnosis

#### `calibration_curve.png`
- **Purpose**: Evaluates how well predicted probabilities match actual outcomes
- **Contents**: Plot comparing predicted probability vs actual frequency
- **Interpretation**:
  - **Perfect calibration**: Diagonal line (45-degree)
  - **Overconfident**: Curve below diagonal
  - **Underconfident**: Curve above diagonal
- **Medical Relevance**: Well-calibrated probabilities crucial for clinical decision-making

## 📋 Report Documents

### `Practice2_report.pdf`
- Comprehensive technical report of the project
- Includes methodology, results, and conclusions
- Academic-style documentation with proper citations

### `main.typ` / `main_latex.typ`
- Source files for generating the technical report
- Written in Typst markup language
- Contains structured analysis and findings

## 🔍 How to Interpret Results

### For Data Scientists:
1. **Start with `data_overview.png`** to understand the dataset
2. **Review `feature_distributions.png`** for preprocessing insights
3. **Analyze `correlation_matrix.png`** for feature relationships
4. **Examine `violin_plots.png`** for class separability

### For Medical Professionals:
1. **Focus on `violin_plots.png`** to see which lab values are most affected
2. **Review `confusion_matrix.png`** for diagnostic accuracy
3. **Check `calibration_curve.png`** for probability reliability
4. **Understand limitations** mentioned in the technical report

### For Model Evaluation:
1. **Training stability**: `training_history.png`
2. **Classification performance**: `confusion_matrix.png`
3. **Probability reliability**: `calibration_curve.png`
4. **Feature importance**: Derived from `violin_plots.png`

## 🚨 Important Notes

- **Educational Purpose**: All analyses are for research and educational use only
- **Clinical Validation**: Results require validation on additional datasets
- **Professional Consultation**: Always consult healthcare professionals for medical decisions
- **Data Privacy**: All data used is publicly available and anonymized

## 📊 Regenerating Figures

To regenerate any of these figures, run the corresponding Jupyter notebooks:
- Data figures: `01-data-exploration.ipynb`
- Model figures: `03-model-training.ipynb`
- Performance analysis: `04-model-prediction.ipynb`

---

*Generated as part of the Hepatitis C Prediction Project by Endika, Igor, and Wassim*