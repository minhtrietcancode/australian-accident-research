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
    top10_severe.to_csv('Task 8 FULL COMBINATION vs SEVERITY/task8_combined_top10_severe.csv', index=False)

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
    plt.savefig('Task 8 FULL COMBINATION vs SEVERITY/task8_combined_top10_severe.png')
    plt.close()

    # 2. Identify top 10 combinations with highest percentage of serious/fatal accidents
    # -------------------------------------------------------------------------------
    # Filter for combinations with sufficient data points (at least 5 accidents)
    filtered_df = combined_df[combined_df['ACCIDENT_COUNT'] >= 5]

    # Sort by SERIOUS_FATAL_PERCENT (descending)
    top10_serious_prob = filtered_df.sort_values('SERIOUS_FATAL_PERCENT', ascending=False).head(10)

    # Save to CSV
    top10_serious_prob.to_csv('Task 8 FULL COMBINATION vs SEVERITY/task8_combined_top10_serious_prob.csv', index=False)

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
    plt.savefig('Task 8 FULL COMBINATION vs SEVERITY/task8_combined_top10_serious_prob.png')
    plt.close()

    print("Task 8.2 completed: Analysis and visualizations have been saved.")


# Run the code
task8_2()