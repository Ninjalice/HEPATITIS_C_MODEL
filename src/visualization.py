import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import pandas as pd

plt.style.use('default')
sns.set_palette("husl")

def plot_data_overview(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    if 'Category' in df.columns:
        df['Category'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Disease Categories')
        axes[0,0].set_xlabel('Category')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
    
    if 'Age' in df.columns:
        df['Age'].hist(bins=20, ax=axes[0,1])
        axes[0,1].set_title('Age Distribution')
        axes[0,1].set_xlabel('Age (years)')
        axes[0,1].set_ylabel('Frequency')
    
    if 'Sex' in df.columns:
        df['Sex'].value_counts().plot(kind='pie', ax=axes[1,0], autopct='%1.1f%%')
        axes[1,0].set_title('Sex Distribution')
    
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        missing.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Missing Values by Column')
        axes[1,1].set_ylabel('Count')
        axes[1,1].tick_params(axis='x', rotation=45)
    else:
        axes[1,1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Missing Values')
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        return plt.gcf()
    else:
        print("Not enough numeric columns for correlation matrix")
        return None

def plot_feature_distributions(df, target_col='target'):
    numeric_cols = ['ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if not available_cols:
        print("No feature columns found")
        return None
    
    n_cols = 5
    n_rows = (len(available_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, feature in enumerate(available_cols):
        if i < len(axes):
            if target_col in df.columns:
                for target_value in df[target_col].unique():
                    subset = df[df[target_col] == target_value][feature]
                    label = 'Healthy' if target_value == 0 else 'Hepatitis C'
                    axes[i].hist(subset, alpha=0.7, label=label, bins=20)
                axes[i].set_title(f'{feature} Distribution')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
            else:
                df[feature].hist(bins=20, ax=axes[i])
                axes[i].set_title(f'{feature} Distribution')
    
    for i in range(len(available_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history['train_loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names=['Healthy', 'Hepatitis C']):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.1, 0.02, f'Overall Accuracy: {accuracy:.3f}', fontsize=12)
    
    plt.tight_layout()
    return plt.gcf()

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_precision_recall_curve(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_prediction_confidence(y_true, y_probs, class_names=['Healthy', 'Hepatitis C']):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    y_pred = np.argmax(y_probs, axis=1)
    max_probs = np.max(y_probs, axis=1)
    
    for class_idx in [0, 1]:
        mask = y_pred == class_idx
        if np.any(mask):
            axes[0].hist(max_probs[mask], bins=20, alpha=0.7, 
                        label=f'Predicted: {class_names[class_idx]}')
    
    axes[0].set_title('Prediction Confidence by Predicted Class')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for class_idx in [0, 1]:
        mask = y_true == class_idx
        if np.any(mask):
            axes[1].hist(max_probs[mask], bins=20, alpha=0.7, 
                        label=f'True: {class_names[class_idx]}')
    
    axes[1].set_title('Prediction Confidence by True Class')
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
