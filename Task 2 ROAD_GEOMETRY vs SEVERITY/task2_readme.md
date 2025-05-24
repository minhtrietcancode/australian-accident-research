# Task 2: Road Geometry vs. Severity Analysis

## Objective

This task explores the relationship between road geometry and accident severity using both categorized and original road geometry columns from the `updated_accident.csv` dataset. The analysis is divided into two parts: an overview visualization and a detailed statistical/proportional analysis.

## Rationale for Column Selection

- **Overview Visualization:**  
  The `CATEGORIZED_ROAD_GEOMETRY` column is used for the initial overview visualization. The original `ROAD_GEOMETRY` column contains too many categories, which would clutter the visualization. Using the categorized column provides a clearer, more concise overview suitable for high-level analysis.

- **Statistical and Proportional Analysis:**  
  For detailed statistics and proportional analysis, the original `ROAD_GEOMETRY` column is used. This ensures that the analysis is unbiased and leverages the specific meanings of each original category, which are important for accurate interpretation.

## Files

-   `task2_2.py`: Generates an overview visualization (e.g., pie chart or bar chart) of accident severity by categorized road geometry.
-   `task2_3.py`: Calculates and visualizes statistics (mean, variance) of severity for each original road geometry category.
-   `task2_4.py`: Calculates and visualizes the proportion of serious/fatal accidents for each original road geometry category.
-   `updated_accident.csv`: Contains accident data, including both `CATEGORIZED_ROAD_GEOMETRY` and `ROAD_GEOMETRY` columns.

## Data Preprocessing Notes

Refer to the [Data Preprocess/README for data preprocess.md](Data Preprocess/README for data preprocess.md) file for details on how the `CATEGORIZED_ROAD_GEOMETRY` column was derived from the original `ROAD_GEOMETRY` column. The categorized column is used only for overview purposes, while the original column is retained for detailed analysis.