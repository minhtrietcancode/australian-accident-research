'''
Task 5.1: Prepare Combined Analysis DataFrame
    Create a function named task5_1() that:
        + Uses column CATEGORIZED_LIGHT_CONDITION from updated_accident.csv
        + Uses column ROAD_GEOMETRY_DESC from updated_accident.csv
        + Generates all possible combinations of these two columns
        + Creates new column "LIGHT_GEOMETRY" with combined values (e.g., "DayLight_Curve")
        + Groups data by the LIGHT_GEOMETRY column
        + Calculates for each group:
            - Severity mean (store in column "SEVERITY_MEAN")
            - Severity variance (store in column "SEVERITY_VARIANCE")
            - Percentage of serious/fatal accidents where SEVERITY is 1 or 2 (store in column "SERIOUS_FATAL_PERCENT")
        + Returns a comprehensive DataFrame with these metrics for all combinations
        + Function signature: def task5_1(): # implementation # return updated_df
'''
import pandas as pd
import numpy as np


def task5_1():
    """
    Creates a DataFrame with combined analysis of light conditions and road geometry.

    Returns:
        DataFrame with metrics for all light condition and road geometry combinations.
    """
    # Load the data
    accident = pd.read_csv('../Data Preprocess/updated_accident.csv')

    # Create combinations of light condition and road geometry
    # First, get unique values of each column
    light_conditions = accident['CATEGORIZED_LIGHT_CONDITION'].unique()
    road_geometries = accident['ROAD_GEOMETRY_DESC'].unique()

    # Create all possible combinations
    combinations = []
    for light in light_conditions:
        for geometry in road_geometries:
            combinations.append({
                'CATEGORIZED_LIGHT_CONDITION': light,
                'ROAD_GEOMETRY_DESC': geometry,
                'LIGHT_GEOMETRY': f"{light}_{geometry}"
            })

    # Create a DataFrame with all possible combinations
    all_combinations_df = pd.DataFrame(combinations)

    # Create combined column in the original DataFrame
    accident['LIGHT_GEOMETRY'] = accident['CATEGORIZED_LIGHT_CONDITION'] + '_' + accident['ROAD_GEOMETRY_DESC']

    # Group by combined column and calculate metrics
    grouped = accident.groupby('LIGHT_GEOMETRY').agg(
        SEVERITY_MEAN=('SEVERITY', 'mean'),
        SEVERITY_VARIANCE=('SEVERITY', 'var'),
        SERIOUS_FATAL_COUNT=('SEVERITY', lambda x: np.sum((x == 1) | (x == 2))),
        TOTAL_COUNT=('SEVERITY', 'count')
    )

    # Calculate percentage of serious/fatal accidents
    grouped['SERIOUS_FATAL_PERCENT'] = (grouped['SERIOUS_FATAL_COUNT'] / grouped['TOTAL_COUNT']) * 100

    # Drop the intermediate columns
    grouped = grouped.drop(columns=['SERIOUS_FATAL_COUNT', 'TOTAL_COUNT'])

    # Merge with all combinations to ensure all combinations are included
    # This preserves CATEGORIZED_LIGHT_CONDITION and ROAD_GEOMETRY_DESC columns needed for Task 5.2
    result = pd.merge(all_combinations_df, grouped, on='LIGHT_GEOMETRY', how='left')

    # Fill NaN values (combinations with no accidents) with 0
    result = result.fillna(0)

    # Ensure the original columns are kept in the final result
    # This is important for the heatmap in Task 5.2
    assert 'CATEGORIZED_LIGHT_CONDITION' in result.columns
    assert 'ROAD_GEOMETRY_DESC' in result.columns

    return result
