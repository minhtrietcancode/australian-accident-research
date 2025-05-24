# Task 7: Combined Analysis of Road Geometry and Road Surface vs. Severity

## Objective

This task investigates the combined effect of road geometry and road surface on accident severity. The analysis aims to provide insights into how different combinations of road configurations and surface types relate to the severity of accidents.

## Rationale for Column Selection

- **Road Geometry:**  
  The original `ROAD_GEOMETRY_DESC` column from `updated_accident.csv` is used. This choice preserves the full detail and objectiveness of the data, as the categorized version would group several distinct configurations into broader categories, potentially hiding important information.

- **Road Surface:**  
  The `CATEGORIZED_ROAD_SURFACE` column from `updated_vehicle.csv` is used. The categorized and original columns for road surface have essentially the same meaning, so using the categorized version provides a cleaner set of categories without any loss of information or introduction of bias.

## Files

- `task7_1.py`: Prepares a DataFrame combining `ROAD_GEOMETRY_DESC` and `CATEGORIZED_ROAD_SURFACE`, calculates severity statistics (mean, variance, and serious/fatal percentage) for each combination, and outputs a comprehensive table for further analysis.
- `updated_accident.csv`: Contains processed accident data with the original road geometry column.
- `updated_vehicle.csv`: Contains processed vehicle data with the categorized road surface column.

## Data Processing Notes

- All possible combinations of road geometry and road surface are considered, even if some combinations do not occur in the data.
- For each combination, the following metrics are calculated:
  - Mean severity (`SEVERITY_MEAN`)
  - Severity variance (`SEVERITY_VARIANCE`)
  - Percentage of serious/fatal accidents (`SERIOUS_FATAL_PERCENT`), where severity is 1 or 2

## Summary

This combined analysis provides a detailed view of how specific road geometry and surface scenarios relate to accident severity, using the most objective and meaningful columns for each factor.