# Feature Engineering and EDA

This repository contains Jupyter notebooks demonstrating various feature engineering and exploratory data analysis (EDA) techniques commonly used in machine learning projects.

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
git clone https://github.com/Abhinavexists/Feature_Engineering.git
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

## Project Structure

```
Feature_Engineering/
├── Exploratory_Data_Analysis/
│   ├── Dataset/
│   ├── Data_Cleaning.ipynb
│   ├── Flight_price.ipynb
│   └── Red_Wine.ipynb
├── Feature_Engineering/
│   ├── Encoding/
│   │   ├── Label_Encoding.ipynb
│   │   ├── OHE_Encoding.ipynb
│   │   ├── Ordinal_Encoding.ipynb
│   │   └── Target_guided_Ordinal_Encoding.ipynb
│   ├── Handle_missing_values.ipynb
│   ├── Handling_and_Outliers.ipynb
│   ├── Handling_Imbalanced_dataset.ipynb
│   └── SMOTE.ipynb
├── LICENSE
├── readme.md
└── requirements.txt
```

## Techniques Covered

### 1. Data Cleaning and EDA
- Data cleaning and preprocessing techniques
- Exploratory data analysis on various datasets:
  - Flight price prediction
  - Google Play Store apps
  - Red wine quality analysis

### 2. Feature Engineering Techniques
- Handling Missing Values
  - Various imputation methods
  - Deletion strategies
- Handling Outliers
  - Detection and treatment methods
  - IQR method implementation
- Handling Imbalanced Datasets
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - Upsampling and downsampling methods

### 3. Encoding Techniques
- Label Encoding
- One-Hot Encoding (OHE)
- Ordinal Encoding
- Target-guided Ordinal Encoding

## Example Use Cases

### 1. Flight Price Prediction
The repository includes a comprehensive example of feature engineering for flight price prediction, including:
- Date and time feature extraction
- Duration calculation
- Handling categorical variables
- Route analysis
- Stop information processing

Features processed include:
- Airline
- Source/Destination cities
- Departure/Arrival times
- Flight duration
- Number of stops
- Additional flight information

### 2. Red Wine Quality Analysis
This dataset focuses on Portuguese "Vinho Verde" red wine variants, analyzing physicochemical properties to predict wine quality. The analysis includes:

#### Dataset Features:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (score between 0 and 10)

#### Analysis Techniques:
- Correlation analysis between wine properties
- Distribution analysis of wine qualities
- Feature relationships visualization
- Outlier detection
- Quality prediction modeling

### Code Examples

#### Feature Extraction from Datetime
```python
# Split date into components
df[['Date','Month','Year']] = df['Date_of_Journey'].str.split('/',expand=True)
df[['Date', 'Month', 'Year']] = df[['Date', 'Month', 'Year']].astype(int)
```

#### Duration Processing
```python
# Convert duration to minutes
df['Duration_Time'] = (df['Duration'].str.extract('(?:(\d+)h)?\s*(?:(\d+)m)?')
                      .fillna(0)
                      .astype(int)
                      .apply(lambda x:x[0]*60 + x[1], axis=1))
```

#### Categorical Encoding
```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
transformed = encoder.fit_transform(df[['Airline','Source','Destination']]).toarray()
```
## Learning Path

For those new to feature engineering, follow this sequence:
1. Data_Cleaning.ipynb → Handle_missing_values.ipynb
2. Handling_and_Outliers.ipynb → Label_Encoding.ipynb
3. Handling_Imbalanced_dataset.ipynb → SMOTE.ipynb
4. Apply techniques in case studies: Flight_price.ipynb and Red_Wine.ipynb

## Troubleshooting

- **Missing Dependencies**: Run `pip install imbalanced-learn` if SMOTE is missing
- **Memory Errors with OHE**: Use `pd.get_dummies(sparse=True)` for high-cardinality columns
- **Jupyter Not Showing Plots**: Add `%matplotlib inline` at notebook top


## Contributing

Feel free to contribute to this repository by:
1. Forking the repository
2. Creating a new branch for your feature
3. Submitting a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Abhinav Kumar Singh
- GitHub: [@Abhinavexists](https://github.com/Abhinavexists)