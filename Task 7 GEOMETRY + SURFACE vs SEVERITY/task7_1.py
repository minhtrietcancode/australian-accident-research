'''
Task 7.1: Prepare Combined Analysis DataFrame
    - Create a function named task7_1() that:
        - Uses column ROAD_GEOMETRY_DESC from updated_accident.csv
        - Uses column CATEGORIZED_ROAD_SURFACE from updated_vehicle.csv
        - Merges the two datasets on ACCIDENT_NO
        - Generates all possible combinations of these two columns
        - Creates new column "GEOMETRY_SURFACE" with combined values (e.g., "Curve_Wet")
        - Groups data by the GEOMETRY_SURFACE column
        - Calculates for each group:
            + Severity mean (store in column "SEVERITY_MEAN")
            + Severity variance (store in column "SEVERITY_VARIANCE")
            + Percentage of serious/fatal accidents where SEVERITY is 1 or 2 (store in column "SERIOUS_FATAL_PERCENT")
        - Returns a comprehensive DataFrame with these metrics for all combinations
        - Function signature: def task7_1(): # implementation # return combined_df

'''
import pandas as pd
import numpy as np


def task7_1():
    """
    Creates a DataFrame with combined analysis of road geometry and road surface.

    Returns:
        DataFrame with metrics for all road geometry and road surface combinations.
    """
    # Load the data
    accident = pd.read_csv('Data Preprocess/updated_accident.csv')
    vehicle = pd.read_csv('Data Preprocess/updated_vehicle.csv')

    # Merge the two datasets on ACCIDENT_NO
    merged_df = pd.merge(accident, vehicle, on='ACCIDENT_NO', how='inner')

    # Create combinations of road geometry and road surface
    # First, get unique values of each column
    road_geometries = accident['ROAD_GEOMETRY_DESC'].unique()
    road_surfaces = vehicle['CATEGORIZED_ROAD_SURFACE'].unique()

    # Create all possible combinations
    combinations = []
    for geometry in road_geometries:
        for surface in road_surfaces:
            combinations.append({
                'ROAD_GEOMETRY_DESC': geometry,
                'CATEGORIZED_ROAD_SURFACE': surface,
                'GEOMETRY_SURFACE': f"{geometry}_{surface}"
            })

    # Create a DataFrame with all possible combinations
    all_combinations_df = pd.DataFrame(combinations)

    # Create combined column in the merged DataFrame
    merged_df['GEOMETRY_SURFACE'] = merged_df['ROAD_GEOMETRY_DESC'] + '_' + merged_df['CATEGORIZED_ROAD_SURFACE']

    # Group by combined column and calculate metrics
    grouped = merged_df.groupby('GEOMETRY_SURFACE').agg(
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
    combined_df = pd.merge(all_combinations_df, grouped, on='GEOMETRY_SURFACE', how='left')

    # Fill NaN values (combinations with no accidents) with 0
    combined_df = combined_df.fillna(0)

    # Ensure the original columns are kept in the final result
    assert 'ROAD_GEOMETRY_DESC' in combined_df.columns
    assert 'CATEGORIZED_ROAD_SURFACE' in combined_df.columns

    return combined_df