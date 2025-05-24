import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the dataset
accident = pd.read_csv('Data Preprocess/updated_accident.csv')

# ------------ Left plot: Mean and Variance of Severity ------------
# Calculate mean and variance of severity for each road geometry category
severity_stats = accident.groupby('ROAD_GEOMETRY_DESC')['SEVERITY'].agg(['mean', 'var']).reset_index()
severity_stats.columns = ['ROAD_GEOMETRY_CATEGORY', 'SEVERITY_MEAN', 'SEVERITY_VARIANCE']

# Sort by SEVERITY_MEAN first, then SEVERITY_VARIANCE (ascending order)
severity_stats = severity_stats.sort_values(by=['SEVERITY_MEAN', 'SEVERITY_VARIANCE'])

# Save statistics to CSV
severity_stats.to_csv('task2_3_stat.csv', index=False)

# ------------ Right plot: Percentage of Serious/Fatal Accidents ------------
# Define function to calculate percentage of serious/fatal accidents
def calculate_serious_accident_prob(df, group_col):
    # Count total accidents per category
    total_accidents = df.groupby(group_col).size()

    # Count serious/fatal accidents (SEVERITY 1-2) per category
    serious_fatal = df[df['SEVERITY'].isin([1, 2])]
    serious_fatal_count = serious_fatal.groupby(group_col).size()

    # Calculate percentage
    serious_fatal_prob = (serious_fatal_count / total_accidents) * 100

    # Create DataFrame with results
    result_df = pd.DataFrame({
        'ROAD_GEOMETRY_CATEGORY': serious_fatal_prob.index,
        'SERIOUS_AND_FATAL_ACCIDENT_PROB': serious_fatal_prob.values
    })

    # Sort by percentage in descending order
    result_df = result_df.sort_values('SERIOUS_AND_FATAL_ACCIDENT_PROB', ascending=False)

    return result_df

# Calculate percentage of serious/fatal accidents for each road geometry category
road_geometry_serious_prob = calculate_serious_accident_prob(accident, 'ROAD_GEOMETRY_DESC')

# Save to CSV
road_geometry_serious_prob.to_csv('task2_4_road_geometry_serious_prob.csv', index=False)

# ------------ Create combined figure ------------
# Create figure with two subplots side by side - reduced overall size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plt.subplots_adjust(wspace=0.2)  # Reduced space between plots

# --- First subplot: Mean and Variance of Severity ---
# Set width of bars
bar_width = 0.15  # Further reduced bar width
index = np.arange(len(severity_stats))

# Create bars on the first axis
mean_bars = ax1.bar(index - bar_width/2, severity_stats['SEVERITY_MEAN'], bar_width,
                   label='Mean', color='#1f77b4', alpha=0.8)

variance_bars = ax1.bar(index + bar_width/2, severity_stats['SEVERITY_VARIANCE'], bar_width,
                       label='Variance', color='#ff7f0e', alpha=0.8)

# Add labels and title for the first subplot
ax1.set_xlabel('Road Geometry', fontsize=9)
ax1.set_ylabel('Value', fontsize=9)
ax1.set_title('Severity: Mean & Variance', fontsize=10)
ax1.set_xticks(index)
ax1.set_xticklabels(severity_stats['ROAD_GEOMETRY_CATEGORY'], rotation=45, ha='right', fontsize=7)
ax1.legend(fontsize=8, loc='upper right', frameon=False)
ax1.grid(axis='y', linestyle='--', alpha=0.2)

# Add values on top of bars with smaller font
for bar in mean_bars:
    height = bar.get_height()
    if height > 0.1:  # Only add text if there's enough space
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=6)

for bar in variance_bars:
    height = bar.get_height()
    if height > 0.1:  # Only add text if there's enough space
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=6)

# --- Second subplot: Percentage of Serious/Fatal Accidents ---
# Sort categories in descending order for the second plot
sorted_data = road_geometry_serious_prob.sort_values('SERIOUS_AND_FATAL_ACCIDENT_PROB', ascending=False)

# Create bars on the second axis with reduced width
prob_bars = ax2.bar(sorted_data['ROAD_GEOMETRY_CATEGORY'], 
                   sorted_data['SERIOUS_AND_FATAL_ACCIDENT_PROB'],
                   width=0.4, color='#2ca02c', alpha=0.8)

# Add labels and title for the second subplot
ax2.set_xlabel('Road Geometry', fontsize=9)
ax2.set_ylabel('Percentage (%)', fontsize=9)
ax2.set_title('Serious/Fatal Accidents Percentage', fontsize=10)
ax2.set_xticklabels(sorted_data['ROAD_GEOMETRY_CATEGORY'], rotation=45, ha='right', fontsize=7)
ax2.grid(axis='y', linestyle='--', alpha=0.2)

# Add value labels with smaller font
for bar in prob_bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.15,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=6)

# Adjust overall layout and save
plt.tight_layout(pad=0.5)  # Reduced padding
plt.savefig('geometry_severity_final.png', dpi=300, bbox_inches='tight')
plt.close()

print("Combined visualization saved successfully as geometry_severity_final.png!")