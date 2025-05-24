'''
Task 5.2: Analysis and Visualization
    + Use the DataFrame returned by function task5_1()
    + Create a heatmap showing mean SEVERITY for each LIGHT_GEOMETRY combination:
        - Use CATEGORIZED_LIGHT_CONDITION on one axis and ROAD_GEOMETRY_DESC on the other
        - Color-code by severity mean (darker colors for lower/more severe values)
        - Save as: task5_light_geometry_heatmap.png
    + Identify top 10 combinations with lowest mean SEVERITY and appropriate variance:
        - Sort the DataFrame by SEVERITY_MEAN (ascending) and SEVERITY_VARIANCE (ascending)
        - Select the top 10 rows
        - Output results as CSV file: task5_light_geometry_top10_severe.csv
        - Create bar chart visualization of these top 10 combinations and save as: task5_light_geometry_top10_severe.png
    + Identify top 10 combinations with highest percentage of serious/fatal accidents:
        - Sort the DataFrame by SERIOUS_FATAL_PERCENT (descending)
        - Select the top 10 rows
        - Output results as CSV file: task5_light_geometry_top10_serious_prob.csv
        - Create bar chart visualization of these top 10 combinations and save as: task5_light_geometry_top10_serious_prob.png

'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from task5_1 import task5_1

def task5_2():
    """
    Performs analysis and visualization on light condition and road geometry data.
    Creates a heatmap and identifies/visualizes top 10 combinations based on different metrics.
    """
    # Get the DataFrame from task5_1
    light_geometry_df = task5_1()

    # 1. Create a heatmap showing mean SEVERITY for each combination
    # ----------------------------------------------------------
    # Pivot the data for the heatmap
    pivot_df = light_geometry_df.pivot(
        index='CATEGORIZED_LIGHT_CONDITION',
        columns='ROAD_GEOMETRY_DESC',
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

    plt.title('Mean Severity by Light Condition and Road Geometry\n(Darker = More Severe)', fontsize=16)
    plt.tight_layout()
    plt.savefig('task5_light_geometry_heatmap.png')
    plt.close()

    # 2. Identify top 10 combinations with lowest mean SEVERITY and appropriate variance
    # ------------------------------------------------------------------------------
    # Sort by SEVERITY_MEAN (ascending) and SEVERITY_VARIANCE (ascending)
    top10_severe = (
        light_geometry_df[light_geometry_df['SEVERITY_MEAN'] != 0]
        .sort_values(['SEVERITY_MEAN', 'SEVERITY_VARIANCE'])
        .head(10)
    ) # remember to exclude the combination with severity 0, that is the combination without any records, that
      # is why it can has mean severity = 0

    # Save to CSV
    top10_severe.to_csv('task5_light_geometry_top10_severe.csv', index=False)

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
        top10_severe['LIGHT_GEOMETRY'],
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

    plt.title('Top 10 Most Severe Light-Geometry Combinations\n(Lower Mean = More Severe)', fontsize=16)
    plt.xlabel('Light Condition - Road Geometry Combination')
    plt.ylabel('Mean Severity (with Standard Deviation)')
    plt.ylim(0, max(top10_severe['SEVERITY_MEAN']) * 1.5)  # Adjust y-axis to accommodate error bars
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('task5_light_geometry_top10_severe.png')
    plt.close()

    # 3. Identify top 10 combinations with highest percentage of serious/fatal accidents
    # ------------------------------------------------------------------------------
    # Sort by SERIOUS_FATAL_PERCENT (descending)
    top10_serious_prob = light_geometry_df.sort_values('SERIOUS_FATAL_PERCENT', ascending=False)
    top10_serious_prob = top10_serious_prob.head(10)

    # Save to CSV
    top10_serious_prob.to_csv('task5_light_geometry_top10_serious_prob.csv', index=False)

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
        top10_serious_prob['LIGHT_GEOMETRY'],
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

    plt.title('Top 10 Light-Geometry Combinations with Highest Percentage of Serious/Fatal Accidents', fontsize=14)
    plt.xlabel('Light Condition - Road Geometry Combination')
    plt.ylabel('Percentage of Serious/Fatal Accidents (%)')
    plt.ylim(0, max(top10_serious_prob['SERIOUS_FATAL_PERCENT']) * 1.2)  # Add some space for labels
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('task5_light_geometry_top10_serious_prob.png')
    plt.close()

    print("Task 5.2 completed: Analysis and visualizations have been saved.")

# Run the code
task5_2()