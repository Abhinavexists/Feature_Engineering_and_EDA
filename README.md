# Feature Engineering and EDA

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Abhinavexists/Feature_Engineering_and_EDA/graphs/commit-activity)

This repository contains comprehensive Jupyter notebooks demonstrating various **feature engineering** and **exploratory data analysis (EDA)** techniques commonly used in machine learning and data science projects.

## Getting Started

### Prerequisites

- **Python 3.7 or higher** (recommended: Python 3.9+)
- **pip** (Python package installer)
- **Jupyter Notebook or JupyterLab**
- **Git** (for cloning the repository)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Abhinavexists/Feature_Engineering_and_EDA.git
```

1. **Navigate to the project directory**

```bash
cd Feature_Engineering_and_EDA
```

1. **Create a virtual environment** (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

1. **Install required packages**

```bash
pip install -r requirements.txt
```

1. **Launch Jupyter Notebook**

```bash
jupyter notebook
# Or for JupyterLab
jupyter lab
```

## Techniques Covered

### 1. Data Cleaning and EDA

Explore comprehensive data cleaning and preprocessing techniques with real-world datasets:

- **Data Cleaning Techniques**
  - Handling corrupted data entries
  - Data type conversions
  - String manipulation and parsing
  - Dealing with inconsistent formatting

- **Exploratory Data Analysis**
  - **Flight Price Prediction**: Time-based feature extraction, route analysis
  - **Google Play Store Apps**: Category analysis, rating distributions, app size analysis
  - **Red Wine Quality**: Physicochemical property analysis, quality prediction

### 2. Feature Engineering Techniques

#### Handling Missing Values

Learn various strategies to deal with missing data:

- **Deletion Methods**: Row/column-wise deletion (when to use and when to avoid)
- **Imputation Techniques**:
  - Mean/Median/Mode imputation (best for normally distributed data)
  - Forward fill and backward fill (time series data)
  - Advanced imputation methods
- **When to use which method**: Decision guidelines based on data distribution and missingness pattern

#### Handling Outliers

Master outlier detection and treatment:

- **Detection Methods**:
  - IQR (Interquartile Range) method
  - Z-score method
  - Visualization techniques (box plots, scatter plots)
- **Treatment Strategies**:
  - Removal vs. Capping
  - Transformation techniques
  - When outliers are actually valuable data points

#### Handling Imbalanced Datasets

Techniques for addressing class imbalance:

- **SMOTE** (Synthetic Minority Over-sampling Technique)
  - When to use SMOTE vs. other methods
  - Implementation details
  - Potential pitfalls
- **Resampling Strategies**:
  - Upsampling minority class
  - Downsampling majority class
  - Combined approaches

### 3. Encoding Techniques

Transform categorical variables into numerical format:

- **Label Encoding**
  - Best for: Ordinal data with inherent order
  - Pros and cons
  - Practical examples

- **One-Hot Encoding (OHE)**
  - Best for: Nominal categorical variables
  - Memory considerations
  - Handling high-cardinality features
  - Sparse matrix optimization

- **Ordinal Encoding**
  - Custom ordering for ranked categories
  - Domain knowledge integration

- **Target-Guided Ordinal Encoding**
  - Advanced technique using target variable
  - Prevents overfitting considerations
  - Cross-validation strategies

## Learning Path

For those new to feature engineering and EDA, follow this recommended sequence:

### **Beginner Path** (Start here if new to data science)

1. **Data_Cleaning.ipynb** - Learn basic data cleaning
2. **Handle_missing_values.ipynb** - Master missing data strategies
3. **Handling_and_Outliers.ipynb** - Detect and treat outliers
4. **Label_Encoding.ipynb** - Simple categorical encoding

### **Intermediate Path** (Build on fundamentals)

1. **OHE_Encoding.ipynb** - One-Hot Encoding techniques
2. **Ordinal_Encoding.ipynb** - Custom category ordering
3. **Handling_Imbalanced_dataset.ipynb** - Class imbalance strategies
4. **SMOTE.ipynb** - Synthetic oversampling

### **Advanced Path** (Real-world applications)

1. **Target_guided_Ordinal_Encoding.ipynb** - Advanced encoding
2. **Flight_price.ipynb** - Complete case study
3. **Red_Wine.ipynb** - Quality prediction project

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
