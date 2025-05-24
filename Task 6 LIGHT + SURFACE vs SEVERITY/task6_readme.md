# Task 6: Combined Analysis of Light Condition and Road Surface vs. Severity

## Objective

This task examines the combined effect of light condition and road surface on accident severity. The analysis aims to reveal how different combinations of lighting and road surface types relate to the severity of accidents.

## Rationale for Column Selection

- **Light Condition & Road Surface:**  
  The columns `CATEGORIZED_LIGHT_CONDITION` (from `updated_accident.csv`) and `CATEGORIZED_ROAD_SURFACE` (from `updated_vehicle.csv`) are used for this analysis. The categorized versions are chosen because their meanings are essentially the same as the original columns, but they provide a cleaner and more consistent set of categories for analysis and visualization. There is no loss of information or introduction of bias by using the categorized columns in this context.

## Files

- `task6_1.py`: Prepares a DataFrame combining `CATEGORIZED_LIGHT_CONDITION` and `CATEGORIZED_ROAD_SURFACE`, calculates severity statistics (mean, variance, and serious/fatal percentage) for each combination, and outputs a comprehensive table for further analysis.
- `updated_accident.csv`: Contains processed accident data with the categorized light condition column.
- `updated_vehicle.csv`: Contains processed vehicle data with the categorized road surface column.

## Data Processing Notes

- All possible combinations of light condition and road surface are considered, even if some combinations do not occur in the data.
- For each combination, the following metrics are calculated:
  - Mean severity (`SEVERITY_MEAN`)
  - Severity variance (`SEVERITY_VARIANCE`)
  - Percentage of serious/fatal accidents (`SERIOUS_FATAL_PERCENT`), where severity is 1 or 2

## Summary

This combined analysis provides a clear view of how specific lighting and road surface scenarios relate to accident severity, using categorized columns that retain the full meaning of the original data.