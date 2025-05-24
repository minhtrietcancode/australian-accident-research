# Task 1: Light Condition vs. Severity Analysis

## Objective

This task provides an overview of the relationship between light conditions and accident severity. It utilizes the `CATEGORIZED_LIGHT_CONDITION` column from the `updated_accident.csv` dataset for analysis and visualization.

## Rationale for Using `CATEGORIZED_LIGHT_CONDITION`

The `CATEGORIZED_LIGHT_CONDITION` column is used throughout Task 1 because it offers a simplified and unbiased categorization of the original `LIGHT_CONDITION` values. This categorization facilitates a clear and easily understandable overview of the connection between lighting conditions and accident severity.

## Files

-   [task1_2.py](Task 1 LIGHT_CONDITION VS SEVERITY/task1_2.py): Generates a pie chart showing the distribution of light conditions, a stacked bar chart showing the distribution of light conditions for each severity level, and a pie chart showing the percentage of serious/fatal accidents by light condition.
-   [task1_3.py](Task 1 LIGHT_CONDITION VS SEVERITY/task1_3.py): Calculates the mean and variance of severity for each light condition category and visualizes the statistics using a bar chart.
-   [task1_4.py](Task 1 LIGHT_CONDITION VS SEVERITY/task1_4.py): Calculates the percentage of serious/fatal accidents for each light condition category and visualizes the results using a bar chart.
-   [updated_accident.csv](Data Preprocess/README for data preprocess.md): Contains accident data, including the `CATEGORIZED_LIGHT_CONDITION` and `SEVERITY` columns.

## Data Preprocessing Notes

Refer to the [Data Preprocess/README for data preprocess.md](Data Preprocess/README for data preprocess.md) file for details on how the `CATEGORIZED_LIGHT_CONDITION` column was derived from the original `LIGHT_CONDITION` column. Specifically, note the handling of "Unknown" values.