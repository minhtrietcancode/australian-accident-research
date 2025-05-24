# Task 3: Road Surface vs. Severity Analysis

## Objective

This task analyzes the relationship between road surface type and accident severity using the road surface columns from the `updated_vehicle.csv` dataset. Both the original and categorized columns are used, but since their values are almost identical, there is no significant difference in their application for this analysis.

## Files

- `task3_2.py`: Generates visualizations (e.g., pie chart or bar chart) of accident severity by road surface type.
- `task3_3.py`: Calculates and visualizes statistics (mean, variance) of severity for each road surface category.
- `task3_4.py`: Calculates and visualizes the proportion of serious/fatal accidents for each road surface category.
- `updated_vehicle.csv`: Contains vehicle data, including both `CATEGORIZED_ROAD_SURFACE` and `ROAD_SURFACE_TYPE_DESC` columns.

## Data Preprocessing Notes

Refer to the [Data Preprocess/README for data preprocess.md](../Data%20Preprocess/README%20for%20data%20preprocess.md) file for details on how the `CATEGORIZED_ROAD_SURFACE` column was derived. For this task, either the original or categorized column can be used interchangeably due to their similarity.