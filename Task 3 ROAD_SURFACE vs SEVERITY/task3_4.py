'''
    - Task to do:
        + Calculate percentage of serious/fatal accidents (SEVERITY 1-2) for each road surface category
          --> sort by percentage in descending order
          --> output as task3_4_road_surface_serious_prob.csv with these columns: ROAD_SURFACE_CATEGORY,
              SERIOUS_AND_FATAL_ACCIDENT_PROB
        + Create a bar chart to compare the above percentage for each road surface category
          --> save as task3_4_road_surface_serious_prob.png
    - Datasets and columns to use: same as task3_2.py
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the datasets
accident = pd.read_csv('../Data Preprocess/updated_accident.csv')
vehicle = pd.read_csv('../Data Preprocess/updated_vehicle.csv')

# Merge datasets on ACCIDENT_NO
merged_data = pd.merge(accident, vehicle, on='ACCIDENT_NO')


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
        'ROAD_SURFACE_CATEGORY': serious_fatal_prob.index,
        'SERIOUS_AND_FATAL_ACCIDENT_PROB': serious_fatal_prob.values
    })

    # Sort by percentage in descending order
    result_df = result_df.sort_values('SERIOUS_AND_FATAL_ACCIDENT_PROB', ascending=False)

    return result_df


# Calculate percentage of serious/fatal accidents for each road surface category
road_surface_serious_prob = calculate_serious_accident_prob(merged_data, 'CATEGORIZED_ROAD_SURFACE')

# Save to CSV
road_surface_serious_prob.to_csv('task3_4_road_surface_serious_prob.csv', index=False)

# Print data to verify
print("Percentage of Serious/Fatal Accidents by Road Surface:")
print(road_surface_serious_prob)

# Create bar chart visualization
plt.figure(figsize=(12, 6))

# Create horizontal bar chart for better readability
categories = road_surface_serious_prob['ROAD_SURFACE_CATEGORY']
probabilities = road_surface_serious_prob['SERIOUS_AND_FATAL_ACCIDENT_PROB']

bars = plt.bar(categories, probabilities, color='#1f77b4')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

# Add labels and title
plt.xlabel('Road Surface Category', fontsize=12)
plt.ylabel('Percentage of Serious/Fatal Accidents (%)', fontsize=12)
plt.title('Percentage of Serious/Fatal Accidents (Severity 1-2) by Road Surface', fontsize=14)

# Rotate x labels if needed for better readability
plt.xticks(rotation=45, ha='right')

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('task3_4_road_surface_serious_prob.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nStatistics and visualization saved successfully!")