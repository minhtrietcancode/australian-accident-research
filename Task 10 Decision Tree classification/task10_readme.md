# Task 10: Decision Tree Classification for Accident Severity Prediction

## Overview
This directory contains the implementation of Decision Tree classification models for predicting accident severity based on road and environmental conditions. The severity is categorized into two classes:
- **SIGNIFICANT**: Severity levels 1 and 2 (more severe accidents)
- **NORMAL**: Severity levels 3 and 4 (less severe accidents)

The models analyze three key features:
- `CATEGORIZED_LIGHT_CONDITION`
- `ROAD_GEOMETRY_DESC`
- `CATEGORIZED_ROAD_SURFACE`

## Model Versions

### Version 1
The first version uses the original categorical features directly:
- `CATEGORIZED_LIGHT_CONDITION`
- `ROAD_GEOMETRY_DESC`
- `CATEGORIZED_ROAD_SURFACE`

This approach demonstrates the ability of decision trees to handle categorical features without preprocessing them into numerical values.

File: `version_1.py`

### Version 2
The second version improves upon the first by creating broader categories for each feature based on accident severity risk:

**Broader Light Condition Categories:**
- Poor Lighting: "Dark without Lighting"
- Adequate Lighting: "Dark with lighting", "Daylight", "Limited Light"
- Unknown Lighting: "Unknown"

**Broader Road Geometry Categories:**
- High Risk Geometry: "Not at intersection", "Dead end"
- Moderate Risk Geometry: "Multiple intersection", "T intersection", "Y intersection", "Cross intersection", "Road closure"
- Low Risk Geometry: "Private property"

**Broader Road Surface Categories:**
- Known Road Surface: "Paved", "Gravel", "Unpaved"
- Unknown Road Surface: "Unknown"

File: `version_2.py`

## Implementation Details

Both versions use:
- 5-fold cross-validation
- Random state seed of 42 for reproducibility
- Decision tree with default parameters
- Visualization of tree structure limited to depth 3 for readability
- Feature importance analysis
- Performance metrics including accuracy scores and confusion matrices

## Directory Structure

```
Task 10 Decision Tree classification/
├── version_1.py                # First version implementation
├── version_2.py                # Second version with broader categories
├── images/                     # Visualizations directory
│   ├── version1/               # Visualizations from version 1
│   └── version2/               # Visualizations from version 2
├── results/                    # Results directory
│   ├── version1/               # Text output results from version 1
│   └── version2/               # Text output results from version 2
└── task10_readme.md            # This README file
```

## Results

The models' performance is documented through:
- Individual fold accuracy scores
- Confusion matrices for each fold
- Average accuracy across all folds
- Average confusion matrix
- Classification reports with precision, recall, and F1 scores
- Feature importance rankings

Both versions include visualizations of:
- Confusion matrices for each fold and average
- Accuracy distributions across folds
- Decision tree structure (limited to depth 3)

## Running the Code

To run either version of the model:

```bash
python version_1.py  # For version 1
python version_2.py  # For version 2
```

Both scripts require the following dependencies:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

The input data is expected to be in:
- `Data Preprocess/updated_accident.csv`
- `Data Preprocess/updated_vehicle.csv`

## Advantages of Decision Trees

Decision trees offer several advantages for this classification task:
1. Ability to handle categorical data directly
2. Easy interpretation of decision rules
3. Feature importance assessment
4. No assumption about data distribution
5. Ability to capture non-linear relationships

## Comparison with k-NN (Task 9)

Unlike the k-NN approach in Task 9, decision trees:
- Do not require feature scaling or normalization
- Provide directly interpretable rules
- Can identify the most important features
- Handle categorical data without conversion to numerical risk scores

Version 2's broader categorization approach simplifies the model while potentially improving generalization by reducing the risk of overfitting to specific subcategories. 