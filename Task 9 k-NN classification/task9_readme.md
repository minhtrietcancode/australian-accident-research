# Task 9: k-NN Classification for Accident Severity Prediction

## Overview
This directory contains the implementation of k-Nearest Neighbors (k-NN) classification models for predicting accident severity based on road conditions. The severity is categorized into two classes:
- **SIGNIFICANT**: Severity levels 1 and 2 (more severe accidents)
- **NORMAL**: Severity levels 3 and 4 (less severe accidents)

## Model Versions

### Version 1
The first version uses risk scores derived from the probability of serious/fatal accidents for each category of the three conditions:
- `SIGNIFICANT_SEVERITY_RISK_LIGHT_CONDITION`
- `SIGNIFICANT_SEVERITY_RISK_ROAD_GEOMETRY`
- `SIGNIFICANT_SEVERITY_RISK_ROAD_SURFACE`

These risk scores are based on the analysis from Tasks 1, 2, and 3, and they represent numerical probability values (scaled to 0-100) for more effective k-NN implementation.

File: `version_1.py`

### Version 2
The second version extends the first approach by incorporating weighted features based on the importance of each condition:
- Light conditions have the highest weight
- Road geometry has moderate weight
- Road surface has the lowest weight

#### Weighting Methodology in Detail

The weighting system is implemented using the variance of the percentage of serious/fatal accidents for each condition:

1. **Variance-Based Weighting Calculation**
   - Calculate the variance of the serious/fatal accident probability for each condition
   - Exclude "Unknown" categories from variance calculation to avoid skewing the results
   - The variance represents the spread of the data - a higher variance indicates the condition has more influence on accident severity
   - Calculate weight as: `weight = condition_variance / sum_of_all_variances`

2. **Calculated Weights**
   - Light Condition: ~0.7125 (highest weight due to highest variance of 51.22)
   - Road Geometry: ~0.2488 (medium weight due to medium variance of 17.90)
   - Road Surface: ~0.0387 (lowest weight due to lowest variance of 2.71)

3. **Implementation**
   ```python
   # Calculate variance for each condition (excluding "Unknown" categories)
   light_condition_variance = light_condition_prob[light_condition_prob['LIGHT_CONDITION_CATEGORY'] != 'Unknown']['SERIOUS_AND_FATAL_ACCIDENT_PROB'].var()
   road_geometry_variance = road_geometry_prob[road_geometry_prob['ROAD_GEOMETRY_CATEGORY'] != 'Unknown']['SERIOUS_AND_FATAL_ACCIDENT_PROB'].var()
   road_surface_variance = road_surface_prob[road_surface_prob['ROAD_SURFACE_CATEGORY'] != 'Unknown']['SERIOUS_AND_FATAL_ACCIDENT_PROB'].var()
   
   # Calculate weights based on variance proportion
   variance_sum = light_condition_variance + road_geometry_variance + road_surface_variance
   light_condition_weight = light_condition_variance / variance_sum
   road_geometry_weight = road_geometry_variance / variance_sum
   road_surface_weight = road_surface_variance / variance_sum
   
   # Apply weights to features
   merged_df['WEIGHTED_RISK_LIGHT_CONDITION'] = merged_df['SIGNIFICANT_SEVERITY_RISK_LIGHT_CONDITION'] * light_condition_weight
   merged_df['WEIGHTED_RISK_ROAD_GEOMETRY'] = merged_df['SIGNIFICANT_SEVERITY_RISK_ROAD_GEOMETRY'] * road_geometry_weight
   merged_df['WEIGHTED_RISK_ROAD_SURFACE'] = merged_df['SIGNIFICANT_SEVERITY_RISK_ROAD_SURFACE'] * road_surface_weight
   ```

4. **Rationale Behind Variance-Based Weighting**
   - Higher variance indicates the condition creates more distinction between accident severity outcomes
   - A condition with high variance means changing its category results in significant changes to accident severity probability
   - This makes it a more sensitive/important predictor for the model to consider
   - The weights proportionally adjust the feature space to give more influence to conditions with higher predictive power

File: `version_2.py`

## Implementation Details

Both versions use:
- 5-fold cross-validation
- Optimal k=200 (determined through experimentation)
- 80/20 train/test split (through the 5-fold CV)
- Performance metrics including accuracy scores and confusion matrices
- Data visualization of results

## Directory Structure

```
Task 9 k-NN classification/
├── version_1.py                     # First version of k-NN model implementation
├── version_2.py                     # Second version with weighted features
├── first kNN version .txt result/   # Text output results from version 1
├── first kNN version image result/  # Visualizations from version 1
├── second kNN version .txt result/  # Text output results from version 2
├── second kNN version image result/ # Visualizations from version 2
└── task9_readme.md                  # This README file
```

## Results

The models' performance is documented through:
- Individual fold accuracy scores
- Confusion matrices for each fold
- Average accuracy across all folds
- Average confusion matrix
- Classification reports with precision, recall, and F1 scores

Both versions include visualizations of confusion matrices and accuracy distributions across folds.

### Performance Comparison
The variance-based weighting approach in Version 2 addresses a limitation of standard k-NN, which treats all features equally regardless of their importance. By incorporating the statistical variance of each condition's impact on accident severity, Version 2 improves classification performance by prioritizing features that show more discriminative power in predicting severity outcomes.

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

## Conclusions

The k-NN model demonstrates the ability to predict accident severity based on road conditions. Version 2 with variance-based weighted features addresses the limitation of the first version by incorporating the statistical importance of each condition. This approach effectively transforms the feature space to give more influence to conditions that show greater variability in accident severity outcomes, making the model more sensitive to the most discriminative features. 