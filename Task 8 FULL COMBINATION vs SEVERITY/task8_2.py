"""
Task 8.1: Prepare Combined Analysis DataFrame
    - Create a function named task8_1() that:
        - Uses column CATEGORIZED_LIGHT_CONDITION from updated_accident.csv
        - Uses column ROAD_GEOMETRY_DESC from updated_accident.csv
        - Uses column CATEGORIZED_ROAD_SURFACE from updated_vehicle.csv
        - Merges the datasets on ACCIDENT_NO
        - Generates all possible combinations of these three columns
        - Creates new column "LIGHT_GEOMETRY_SURFACE" with combined values
        - Groups data by the LIGHT_GEOMETRY_SURFACE column
        - Calculates for each group:
            + Severity mean (store in column "SEVERITY_MEAN")
            + Severity variance (store in column "SEVERITY_VARIANCE")
            + Percentage of serious/fatal accidents where SEVERITY is 1 or 2 (store in column "SERIOUS_FATAL_PERCENT")
            + Count of accidents in each combination (store in column "ACCIDENT_COUNT")
        - Returns a comprehensive DataFrame with these metrics for all combinations
"""
import pandas as pd
import numpy as np


def task8_1():
    """
    Creates a DataFrame with combined analysis of light condition, road geometry, and road surface.

    Returns:
        DataFrame with metrics for all light condition, road geometry, and road surface combinations.
    """
    # Load the data
    accident = pd.read_csv('../Data Preprocess/updated_accident.csv')
    vehicle = pd.read_csv('../Data Preprocess/updated_vehicle.csv')

    # Merge the two datasets on ACCIDENT_NO
    merged_df = pd.merge(accident, vehicle, on='ACCIDENT_NO', how='inner')

    # Create combinations of light condition, road geometry, and road surface
    # First, get unique values of each column
    light_conditions = accident['CATEGORIZED_LIGHT_CONDITION'].unique()
    road_geometries = accident['ROAD_GEOMETRY_DESC'].unique()
    road_surfaces = vehicle['CATEGORIZED_ROAD_SURFACE'].unique()

    # Create all possible combinations
    combinations = []
    for light in light_conditions:
        for geometry in road_geometries:
            for surface in road_surfaces:
                combinations.append({
                    'CATEGORIZED_LIGHT_CONDITION': light,
                    'ROAD_GEOMETRY_DESC': geometry,
                    'CATEGORIZED_ROAD_SURFACE': surface,
                    'LIGHT_GEOMETRY_SURFACE': f"{light}_{geometry}_{surface}"
                })

    # Create a DataFrame with all possible combinations
    all_combinations_df = pd.DataFrame(combinations)

    # Create combined column in the merged DataFrame
    merged_df['LIGHT_GEOMETRY_SURFACE'] = (
            merged_df['CATEGORIZED_LIGHT_CONDITION'] + '_' +
            merged_df['ROAD_GEOMETRY_DESC'] + '_' +
            merged_df['CATEGORIZED_ROAD_SURFACE']
    )

    # Group by combined column and calculate metrics
    grouped = merged_df.groupby('LIGHT_GEOMETRY_SURFACE').agg(
        SEVERITY_MEAN=('SEVERITY', 'mean'),
        SEVERITY_VARIANCE=('SEVERITY', 'var'),
        SERIOUS_FATAL_COUNT=('SEVERITY', lambda x: np.sum((x == 1) | (x == 2))),
        ACCIDENT_COUNT=('SEVERITY', 'count')
    )

    # Calculate percentage of serious/fatal accidents
    grouped['SERIOUS_FATAL_PERCENT'] = (grouped['SERIOUS_FATAL_COUNT'] / grouped['ACCIDENT_COUNT']) * 100

    # Drop the intermediate column
    grouped = grouped.drop(columns=['SERIOUS_FATAL_COUNT'])

    # Merge with all combinations to ensure all combinations are included
    combined_df = pd.merge(all_combinations_df, grouped, on='LIGHT_GEOMETRY_SURFACE', how='left')

    # Fill NaN values (combinations with no accidents) with 0
    combined_df = combined_df.fillna(0)

    # Ensure the original columns are kept in the final result
    assert 'CATEGORIZED_LIGHT_CONDITION' in combined_df.columns
    assert 'ROAD_GEOMETRY_DESC' in combined_df.columns
    assert 'CATEGORIZED_ROAD_SURFACE' in combined_df.columns

    return combined_df


"""
Task 8.2: Analysis and Visualization
    1. Identify top 10 combinations with lowest mean SEVERITY and appropriate variance:
        - Sort the DataFrame by SEVERITY_MEAN (ascending) and SEVERITY_VARIANCE (ascending)
        - Filter to include only combinations with sufficient data points (e.g., ACCIDENT_COUNT >= 5)
        - Select the top 10 rows
        - Output results as CSV file: task8_combined_top10_severe.csv
        - Create bar chart visualization of these top 10 combinations and save as: task8_combined_top10_severe.png

    2. Identify top 10 combinations with highest percentage of serious/fatal accidents:
        - Sort the DataFrame by SERIOUS_FATAL_PERCENT (descending)
        - Filter to include only combinations with sufficient data points (e.g., ACCIDENT_COUNT >= 5)
        - Select the top 10 rows
        - Output results as CSV file: task8_combined_top10_serious_prob.csv
        - Create bar chart visualization of these top 10 combinations and save as: task8_combined_top10_serious_prob.png
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from task8_1 import task8_1


def task8_2():
    """
    Performs analysis and visualization on the combined data of light condition,
    road geometry, and road surface.
    Identifies and visualizes top 10 combinations based on different metrics.
    """
    # Get the DataFrame from task8_1
    combined_df = task8_1()

    # 1. Identify top 10 combinations with lowest mean SEVERITY and appropriate variance
    # -------------------------------------------------------------------------------
    # Filter for combinations with sufficient data points (at least 5 accidents)
    filtered_df = combined_df[combined_df['ACCIDENT_COUNT'] >= 5]

    # Sort by SEVERITY_MEAN (ascending) and SEVERITY_VARIANCE (ascending)
    top10_severe = (
        filtered_df[filtered_df['SEVERITY_MEAN'] > 0]  # Exclude combinations with severity 0
        .sort_values(['SEVERITY_MEAN', 'SEVERITY_VARIANCE'])
        .head(10)
    )

    # Save to CSV
    top10_severe.to_csv('task8_combined_top10_severe.csv', index=False)

    # Create bar chart visualization
    plt.figure(figsize=(16, 10))

    # Plot bars with error bars showing variance
    bars = plt.bar(
        range(len(top10_severe)),
        top10_severe['SEVERITY_MEAN'],
        yerr=np.sqrt(top10_severe['SEVERITY_VARIANCE']),  # Standard deviation as error bars
        color='skyblue',
        capsize=7
    )

    # Customize x-axis labels
    plt.xticks(
        range(len(top10_severe)),
        top10_severe['LIGHT_GEOMETRY_SURFACE'],
        rotation=45,
        ha='right'
    )

    # Add labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            f'{top10_severe["SEVERITY_MEAN"].iloc[i]:.2f}\n(n={int(top10_severe["ACCIDENT_COUNT"].iloc[i])})',
            ha='center',
            fontsize=9
        )

    plt.title('Top 10 Most Severe Light-Geometry-Surface Combinations\n(Lower Mean = More Severe)', fontsize=16)
    plt.xlabel('Light Condition - Road Geometry - Road Surface Combination')
    plt.ylabel('Mean Severity (with Standard Deviation)')
    plt.ylim(0, max(top10_severe['SEVERITY_MEAN']) * 1.5)  # Adjust y-axis to accommodate error bars
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('task8_combined_top10_severe.png')
    plt.close()

    # 2. Identify top 10 combinations with highest percentage of serious/fatal accidents
    # -------------------------------------------------------------------------------
    # Filter for combinations with sufficient data points (at least 5 accidents)
    filtered_df = combined_df[combined_df['ACCIDENT_COUNT'] >= 5]

    # Sort by SERIOUS_FATAL_PERCENT (descending)
    top10_serious_prob = filtered_df.sort_values('SERIOUS_FATAL_PERCENT', ascending=False).head(10)

    # Save to CSV
    top10_serious_prob.to_csv('task8_combined_top10_serious_prob.csv', index=False)

    # Create bar chart visualization
    plt.figure(figsize=(16, 10))

    # Plot bars
    bars = plt.bar(
        range(len(top10_serious_prob)),
        top10_serious_prob['SERIOUS_FATAL_PERCENT'],
        color='salmon'
    )

    # Customize x-axis labels
    plt.xticks(
        range(len(top10_serious_prob)),
        top10_serious_prob['LIGHT_GEOMETRY_SURFACE'],
        rotation=45,
        ha='right'
    )

    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f'{top10_serious_prob["SERIOUS_FATAL_PERCENT"].iloc[i]:.2f}%\n(n={int(top10_serious_prob["ACCIDENT_COUNT"].iloc[i])})',
            ha='center',
            fontsize=9
        )

    plt.title('Top 10 Light-Geometry-Surface Combinations with Highest Percentage of Serious/Fatal Accidents',
              fontsize=14)
    plt.xlabel('Light Condition - Road Geometry - Road Surface Combination')
    plt.ylabel('Percentage of Serious/Fatal Accidents (%)')
    plt.ylim(0, max(top10_serious_prob['SERIOUS_FATAL_PERCENT']) * 1.2)  # Add some space for labels
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('task8_combined_top10_serious_prob.png')
    plt.close()

    print("Task 8.2 completed: Analysis and visualizations have been saved.")


# Run the code
task8_2()