'''
    - Task to do:
        + Calculate mean and variance of severity for each road geometry category
          --> sort by SEVERITY_MEAN first, and then SEVERITY_VARIANCE later
          --> Output as: task2_3_stat.csv with columns ROAD_GEOMETRY_CATEGORY, SEVERITY_MEAN, SEVERITY_VARIANCE
        + Create a visualization for the above statistic: bar chart to compare the mean and variance for each
          road geometry category, like each category will have 2 bar associate with it, one bar for mean and one bar
          for variance
          --> save as task2_3_stat.png
    - Datasets and columns to use: same as task2_2.py except for the fact that we will use the original columns as
      when it comes to this part, it is no longer overview as the task2_2.py so cannot use this broader range, we
      can potentially lose some information if we use the broader range like in task2_2.py
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the dataset
accident = pd.read_csv('Data Preprocess/updated_accident.csv')

# Calculate mean and variance of severity for each road geometry category
severity_stats = accident.groupby('ROAD_GEOMETRY_DESC')['SEVERITY'].agg(['mean', 'var']).reset_index()
severity_stats.columns = ['ROAD_GEOMETRY_CATEGORY', 'SEVERITY_MEAN', 'SEVERITY_VARIANCE']

# Sort by SEVERITY_MEAN first, then SEVERITY_VARIANCE (ascending order)
severity_stats = severity_stats.sort_values(by=['SEVERITY_MEAN', 'SEVERITY_VARIANCE'])

# Save statistics to CSV
severity_stats.to_csv('Task 2 ROAD_GEOMETRY vs SEVERITY/task2_3_stat.csv', index=False)

# Print stats to verify
print("Severity Statistics by Road Geometry:")
print(severity_stats)

# Create visualization
plt.figure(figsize=(12, 8))

# Set width of bars
bar_width = 0.35
index = np.arange(len(severity_stats))

# Create bars
mean_bars = plt.bar(index - bar_width / 2, severity_stats['SEVERITY_MEAN'], bar_width,
                    label='Mean Severity', color='#1f77b4', alpha=0.8)

variance_bars = plt.bar(index + bar_width / 2, severity_stats['SEVERITY_VARIANCE'], bar_width,
                        label='Severity Variance', color='#ff7f0e', alpha=0.8)

# Add labels, title and legend
plt.xlabel('Road Geometry Category', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Mean and Variance of Severity by Road Geometry', fontsize=14)
plt.xticks(index, severity_stats['ROAD_GEOMETRY_CATEGORY'], rotation=45, ha='right')
plt.legend()

# Add values on top of bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)

add_labels(mean_bars)
add_labels(variance_bars)

# Adjust layout and save figure
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.savefig('Task 2 ROAD_GEOMETRY vs SEVERITY/task2_3_stat.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nStatistics and visualization saved successfully!")
