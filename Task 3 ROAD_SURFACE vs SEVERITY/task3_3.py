'''
    - Task to do
        + Calculate mean and variance of severity for each road surface category
          --> Output statistics as: task3_3_stat.csv with columns ROAD_SURFACE_CATEGORY, SEVERITY_MEAN, SEVERITY_VARIANCE
          remember to Sort by SEVERITY_MEAN first, then SEVERITY_VARIANCE (ascending order)
        + Create a visualization for the above statistic: bar chart to compare the mean and variance for each
          road surface category, like each category will have 2 bar associate with it, one bar for mean and one bar
          for variance
          --> save as task3_3_stat.png

    - Datasets and notes on columns to use: same as task3_2.py
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the datasets
accident = pd.read_csv('Data Preprocess/updated_accident.csv')
vehicle = pd.read_csv('Data Preprocess/updated_vehicle.csv')

# Merge datasets on ACCIDENT_NO
merged_data = pd.merge(accident, vehicle, on='ACCIDENT_NO')

# Calculate mean and variance of severity for each road surface category
severity_stats = merged_data.groupby('CATEGORIZED_ROAD_SURFACE')['SEVERITY'].agg(['mean', 'var']).reset_index()
severity_stats.columns = ['ROAD_SURFACE_CATEGORY', 'SEVERITY_MEAN', 'SEVERITY_VARIANCE']

# Sort by SEVERITY_MEAN first, then SEVERITY_VARIANCE (ascending order)
severity_stats = severity_stats.sort_values(by=['SEVERITY_MEAN', 'SEVERITY_VARIANCE'])

# Save statistics to CSV
severity_stats.to_csv('Task 3 ROAD_SURFACE vs SEVERITY/task3_3_stat.csv', index=False)

# Print stats to verify
print("Severity Statistics by Road Surface:")
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
plt.xlabel('Road Surface Category', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Mean and Variance of Severity by Road Surface', fontsize=14)
plt.xticks(index, severity_stats['ROAD_SURFACE_CATEGORY'], rotation=45, ha='right')
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
plt.savefig('Task 3 ROAD_SURFACE vs SEVERITY/task3_3_stat.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nStatistics and visualization saved successfully!")