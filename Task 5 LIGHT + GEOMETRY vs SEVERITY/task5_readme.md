# Task 5: Combined Analysis of Light Condition and Road Geometry vs. Severity

## Objective

This task analyzes the combined effect of light condition and road geometry on accident severity. The goal is to understand how different combinations of lighting and road configurations relate to the severity of accidents.

## Rationale for Column Selection

- **Light Condition:**  
  The `CATEGORIZED_LIGHT_CONDITION` column is used because it provides a simplified and unbiased categorization of lighting conditions. This makes the analysis clearer without introducing bias, as the grouping is logical and does not distort the original meaning of the data.

- **Road Geometry:**  
  The original `ROAD_GEOMETRY_DESC` column is used instead of the categorized version. The categorized road geometry column groups several distinct configurations into broader categories, which can hide important details and reduce the objectiveness of the analysis. Using the original column preserves the full granularity and meaning of the data.

## Files

- `task5_1.py`: Prepares a DataFrame combining `CATEGORIZED_LIGHT_CONDITION` and `ROAD_GEOMETRY_DESC`, calculates severity statistics (mean, variance, and serious/fatal percentage) for each combination, and outputs a comprehensive table for further analysis.
- `updated_accident.csv`: Contains the processed accident data with both the categorized light condition and original road geometry columns.

## Data Processing Notes

- All possible combinations of light condition and road geometry are considered, even if some combinations do not occur in the data.
- For each combination, the following metrics are calculated:
  - Mean severity (`SEVERITY_MEAN`)
  - Severity variance (`SEVERITY_VARIANCE`)
  - Percentage of serious/fatal accidents (`SERIOUS_FATAL_PERCENT`), where severity is 1 or 2

## Summary

This combined analysis provides a detailed view of how specific lighting and road geometry scenarios relate to accident severity, using the most objective and meaningful columns for each factor.