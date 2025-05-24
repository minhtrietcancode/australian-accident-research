# Task 8: Full Combination Analysis (Light Condition + Road Geometry + Road Surface) vs. Severity

## Objective

This task analyzes the combined effect of light condition, road geometry, and road surface on accident severity. The goal is to provide a comprehensive view of how specific scenarios involving all three factors relate to the severity of accidents.

## Rationale for Column Selection

- **Light Condition:**  
  The `CATEGORIZED_LIGHT_CONDITION` column from `updated_accident.csv` is used. This categorized version is chosen because it simplifies the data without introducing bias, and its meaning is consistent with the original column.

- **Road Geometry:**  
  The original `ROAD_GEOMETRY_DESC` column from `updated_accident.csv` is used. This preserves the full detail and objectiveness of the data, as the categorized version would group several distinct configurations into broader categories, potentially hiding important information.

- **Road Surface:**  
  The `CATEGORIZED_ROAD_SURFACE` column from `updated_vehicle.csv` is used. The categorized and original columns for road surface have essentially the same meaning, so using the categorized version provides a cleaner set of categories without any loss of information or introduction of bias.

## Files

- `task8_1.py`: Prepares a DataFrame combining `CATEGORIZED_LIGHT_CONDITION`, `ROAD_GEOMETRY_DESC`, and `CATEGORIZED_ROAD_SURFACE`, calculates severity statistics (mean, variance, serious/fatal percentage, and accident count) for each combination, and outputs a comprehensive table for further analysis.
- `updated_accident.csv`: Contains processed accident data with the categorized light condition and original road geometry columns.
- `updated_vehicle.csv`: Contains processed vehicle data with the categorized road surface column.

## Data Processing Notes

- All possible combinations of light condition, road geometry, and road surface are considered, even if some combinations do not occur in the data.
- For each combination, the following metrics are calculated:
  - Mean severity (`SEVERITY_MEAN`)
  - Severity variance (`SEVERITY_VARIANCE`)
  - Percentage of serious/fatal accidents (`SERIOUS_FATAL_PERCENT`), where severity is 1 or 2
  - Count of accidents in each combination (`ACCIDENT_COUNT`)

## Summary

This full combination analysis provides the most detailed view of how specific scenarios involving light condition, road geometry, and road surface relate to accident severity, using the most objective and meaningful columns for each factor.