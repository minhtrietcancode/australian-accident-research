# Task 4: Correlation Analysis

## Objective

This task aims to analyze the relationship between accident severity and three key factors: light condition, road geometry, and road surface. The goal is to determine which factor, or combination of factors, has the strongest relationship with accident severity.

## Implementation Choices

- **Use of Original Encoded Columns:**  
  For correlation analysis, the original encoded columns for each factor are used. This approach avoids introducing bias that could result from categorization, ensuring the most objective calculation of mutual information scores between each factor and SEVERITY.

- **Choice of Correlation Metric:**  
  Mutual Information Correlation is used to measure the relationship between each factor (and their combinations) and SEVERITY. This method is chosen because:
  - The purpose is to compare the influence and connection of each factor with SEVERITY.
  - The categories for each factor do not have a natural order, so assigning ordinal values would be arbitrary and potentially misleading.
  - Mutual information can handle categorical variables without assuming any order.

- **Data Context:**  
  All mutual information calculations are performed on the merged version of `accident.csv` and `vehicle.csv`. This ensures a consistent context for comparing the influence of each factor or combination of factors on SEVERITY.

- **Multivariable Mutual Information:**  
  For combinations of factors, multivariable mutual information is calculated. Each factor in a combination is treated as a separate variable contributing to SEVERITY, rather than combining them into a single composite variable.

- **Interpreting Scores:**  
  The normalized mutual information scores are generally small (around 0.01 or less). This is expected, as many factors contribute to accident severity beyond the three analyzed here. The focus should be on the relative scale of the scores when comparing factors, not their absolute values.

## Files

- `task4_correlation.py`: Performs the mutual information correlation analysis and generates visualizations comparing the influence of each factor and their combinations on SEVERITY.
- `accident.csv` and `vehicle.csv`: Original datasets used for merging and analysis.

## Summary

This correlation analysis provides an objective comparison of how light condition, road geometry, and road surface (individually and in combination) relate to accident severity, using mutual information as the metric.