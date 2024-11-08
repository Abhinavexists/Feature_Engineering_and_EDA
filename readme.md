# Feature Engineering Techniques

This repository contains Jupyter notebooks demonstrating various feature engineering techniques commonly used in machine learning projects.

## Overview

Feature engineering is a crucial step in the machine learning pipeline that involves transforming raw data into features that better represent the underlying problem to predictive models, resulting in improved model accuracy on unseen data.

## Getting Started

### Prerequisites

1. Python 3.7 or higher
2. pip (Python package installer)
3. Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository
```bash
git clone https://github.com/Abhinavks1405/Feature_Engineering.git
```

2. Navigate to the project directory
```bash
cd Feature_Engineering
```

3. Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

4. Install required packages
```bash
pip install -r requirements.txt
```

### Running the Notebooks

1. Start Jupyter Notebook
```bash
jupyter notebook
```

2. Navigate to the desired notebook in your browser
3. Click on the notebook to open it
4. Run the cells using Shift + Enter or the Run button

### Dependencies

Create a `requirements.txt` file in your repository with the following contents:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.2
imbalanced-learn>=0.8.0
seaborn>=0.11.2
matplotlib>=3.4.3
jupyter>=1.0.0
```

## Techniques Covered

### 1. Handling Imbalanced Datasets
- Implementation of different techniques to handle imbalanced datasets:
  - Upsampling (Random Over Sampling)
  - Downsampling (Random Under Sampling)
  - SMOTE (Synthetic Minority Over-sampling Technique)

### 2. Data Encoding Techniques
- Various methods to convert categorical variables into numerical format:
  - Label Encoding
    - Simple numerical encoding for categorical variables
    - Suitable for ordinal data
  - Ordinal Encoding
    - Numerical encoding preserving the order of categories
    - Ideal for ordered categorical data (e.g., education levels)
  - One-Hot Encoding (OHE)
    - Binary encoding for nominal categorical variables
    - Creates dummy variables for each category

### 3. Handling Missing Values
- Understanding different types of missing data:
  - MCAR (Missing Completely at Random)
  - MAR (Missing at Random)
  - MNAR (Missing Not at Random)
- Techniques for handling missing values:
  - Deletion Methods
    - Row-wise deletion
    - Column-wise deletion
  - Imputation Methods
    - Mean imputation (for normally distributed data)
    - Median imputation (for data with outliers)
    - Mode imputation (for categorical variables)

### 4. Handling Outliers
- Five-number summary analysis
- Box plot visualization
- IQR (Interquartile Range) method
- Outlier detection using:
  - Lower fence (Q1 - 1.5 * IQR)
  - Upper fence (Q3 + 1.5 * IQR)

## Repository Structure

```
Feature_Engineering/
├── notebooks/
│   ├── Handling_Imbalanced_dataset.ipynb
│   ├── Label_Encoding.ipynb
│   ├── OHE_Encoding.ipynb
│   ├── Ordinal_Encoding.ipynb
│   ├── SMOTE.ipynb
│   ├── Handle_missing_values.ipynb
│   └── Handling_and_Outliers.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Usage Examples

### Handling Imbalanced Dataset
```python
from sklearn.utils import resample

# Upsampling example
df_minority_upsampled = resample(df_minority,
                                replace=True,
                                n_samples=len(df_majority),
                                random_state=42)
```

### Handling Missing Values
```python
# Mean imputation
df['Age_mean'] = df['age'].fillna(df['age'].mean())

# Median imputation
df['Age_median'] = df['age'].fillna(df['age'].median())

# Mode imputation for categorical variables
mode_value = df['category'].mode()[0]
df['category_filled'] = df['category'].fillna(mode_value)
```

### Detecting Outliers
```python
import numpy as np

# Calculate quartiles and IQR
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

# Define outlier boundaries
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
```

## Troubleshooting

Common issues and solutions:

1. Package installation errors:
```bash
# If you encounter SSL errors
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# If you need to upgrade pip
python -m pip install --upgrade pip
```

2. Jupyter Notebook not starting:
```bash
# Make sure jupyter is installed
pip install jupyter

# Try running with specific port
jupyter notebook --port=8889
```

3. Import errors:
- Make sure all requirements are installed
- Check if virtual environment is activated
- Restart the kernel in Jupyter Notebook

## Contributing

Feel free to contribute to this repository by:
1. Forking the repository
2. Creating a new branch for your feature
```bash
git checkout -b feature/NewFeature
```
3. Committing your changes
```bash
git commit -m "Add new feature"
```
4. Pushing to your branch
```bash
git push origin feature/NewFeature
```
5. Creating a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Abhinav Kumar Singh

## Connect with Me

- GitHub: [@Abhinavks1405](https://github.com/Abhinavks1405)