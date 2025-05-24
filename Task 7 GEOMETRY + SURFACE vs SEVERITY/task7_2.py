'''
Task 7.2: Analysis and Visualization
    - Use the DataFrame returned by function task7_1()
    - Create a heatmap showing mean SEVERITY for each GEOMETRY_SURFACE combination:
        - Use ROAD_GEOMETRY_DESC on one axis and CATEGORIZED_ROAD_SURFACE on the other
        - Color-code by severity mean (darker colors for lower/more severe values)
          --> Save as: task7_geometry_surface_heatmap.png
    - Identify top 10 combinations with lowest mean SEVERITY and appropriate variance:
        - Sort the DataFrame by SEVERITY_MEAN (ascending) and SEVERITY_VARIANCE (ascending)
        - Select the top 10 rows
        - Output results as CSV file: task7_geometry_surface_top10_severe.csv
        - Create bar chart visualization of these top 10 combinations and save as: task7_geometry_surface_top10_severe.png
    Identify top 10 combinations with highest percentage of serious/fatal accidents:
        - Sort the DataFrame by SERIOUS_FATAL_PERCENT (descending)
        - Select the top 10 rows
        - Output results as CSV file: task7_geometry_surface_top10_serious_prob.csv
        - Create bar chart visualization of these top 10 combinations and save as: task7_geometry_surface_top10_serious_prob.png

'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from task7_1 import task7_1

def task7_2():
    """
    Performs analysis and visualization on road geometry and road surface data.
    Creates a heatmap and identifies/visualizes top 10 combinations based on different metrics.
    """
    # Get the DataFrame from task7_1
    geometry_surface_df = task7_1()

    # 1. Create a heatmap showing mean SEVERITY for each combination
    # ----------------------------------------------------------
    # Pivot the data for the heatmap
    pivot_df = geometry_surface_df.pivot(
        index='ROAD_GEOMETRY_DESC',
        columns='CATEGORIZED_ROAD_SURFACE',
        values='SEVERITY_MEAN'
    )

    # Create the heatmap - darker colors for lower/more severe values
    plt.figure(figsize=(14, 10))

    # Using a colormap where darker colors represent lower (more severe) values
    # Lower severity values (1, 2) are more severe than higher values (3, 4)
    sns.heatmap(
        pivot_df,
        cmap='YlOrRd_r',  # Reverse color map so darker colors = lower values (more severe)
        annot=True,  # Show values in cells
        fmt='.2f',  # Format as 2 decimal places
        linewidths=0.5  # Add grid lines
    )

    plt.title('Mean Severity by Road Geometry and Road Surface\n(Darker = More Severe)', fontsize=16)
    plt.tight_layout()
    plt.savefig('task7_geometry_surface_heatmap.png')
    plt.close()

    # 2. Identify top 10 combinations with lowest mean SEVERITY and appropriate variance
    # ------------------------------------------------------------------------------
    # Sort by SEVERITY_MEAN (ascending) and SEVERITY_VARIANCE (ascending)
    top10_severe = (
        geometry_surface_df[geometry_surface_df['SEVERITY_MEAN'] != 0]
        .sort_values(['SEVERITY_MEAN', 'SEVERITY_VARIANCE'])
        .head(10)
    ) # Exclude combinations with severity 0 (those without any records)

    # Save to CSV
    top10_severe.to_csv('task7_geometry_surface_top10_severe.csv', index=False)

    # Create bar chart visualization
    plt.figure(figsize=(14, 8))

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
        top10_severe['GEOMETRY_SURFACE'],
        rotation=45,
        ha='right'
    )

    # Add labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            f'{top10_severe["SEVERITY_MEAN"].iloc[i]:.2f}',
            ha='center',
            fontsize=9
        )

    plt.title('Top 10 Most Severe Geometry-Surface Combinations\n(Lower Mean = More Severe)', fontsize=16)
    plt.xlabel('Road Geometry - Road Surface Combination')
    plt.ylabel('Mean Severity (with Standard Deviation)')
    plt.ylim(0, max(top10_severe['SEVERITY_MEAN']) * 1.5)  # Adjust y-axis to accommodate error bars
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('task7_geometry_surface_top10_severe.png')
    plt.close()

    # 3. Identify top 10 combinations with highest percentage of serious/fatal accidents
    # ------------------------------------------------------------------------------
    # Sort by SERIOUS_FATAL_PERCENT (descending)
    top10_serious_prob = geometry_surface_df.sort_values('SERIOUS_FATAL_PERCENT', ascending=False)
    top10_serious_prob = top10_serious_prob.head(10)

    # Save to CSV
    top10_serious_prob.to_csv('task7_geometry_surface_top10_serious_prob.csv', index=False)

    # Create bar chart visualization
    plt.figure(figsize=(14, 8))

    # Plot bars
    bars = plt.bar(
        range(len(top10_serious_prob)),
        top10_serious_prob['SERIOUS_FATAL_PERCENT'],
        color='salmon'
    )

    # Customize x-axis labels
    plt.xticks(
        range(len(top10_serious_prob)),
        top10_serious_prob['GEOMETRY_SURFACE'],
        rotation=45,
        ha='right'
    )

    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f'{top10_serious_prob["SERIOUS_FATAL_PERCENT"].iloc[i]:.2f}%',
            ha='center',
            fontsize=9
        )

    plt.title('Top 10 Geometry-Surface Combinations with Highest Percentage of Serious/Fatal Accidents', fontsize=14)
    plt.xlabel('Road Geometry - Road Surface Combination')
    plt.ylabel('Percentage of Serious/Fatal Accidents (%)')
    plt.ylim(0, max(top10_serious_prob['SERIOUS_FATAL_PERCENT']) * 1.2)  # Add some space for labels
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('task7_geometry_surface_top10_serious_prob.png')
    plt.close()

    print("Task 7.2 completed: Analysis and visualizations have been saved.")

# Run the code
task7_2()