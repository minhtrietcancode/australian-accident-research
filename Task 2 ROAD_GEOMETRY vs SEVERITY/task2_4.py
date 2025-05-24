'''
    - Task to do:
        + Calculate percentage of serious/fatal accidents (SEVERITY 1-2) for each road geometry category
          --> sort by percentage in descending order
          --> Output as: task2_4_road_geometry_serious_prob.csv with columns for ROAD_GEOMETRY_CATEGORY,
              SERIOUS_AND_FATAL_ACCIDENT_PROB
        + Create a bar chart to compare the above percentage for each road geometry category
          --> save as task2_4_road_geometry_serious_prob.png

    - Dataset to use and notes: same as task2_2.py except for the fact that we will use the original columns as
      when it comes to this part, it is no longer overview as the task2_2.py so cannot use this broader range, we
      can potentially lose some information if we use the broader range like in task2_2.py
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the dataset
accident = pd.read_csv('../Data Preprocess/updated_accident.csv')


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

# Print data to verify
print("Percentage of Serious/Fatal Accidents by Road Geometry:")
print(road_geometry_serious_prob)

# Create bar chart visualization
plt.figure(figsize=(12, 6))

# Create horizontal bar chart for better readability
categories = road_geometry_serious_prob['ROAD_GEOMETRY_CATEGORY']
probabilities = road_geometry_serious_prob['SERIOUS_AND_FATAL_ACCIDENT_PROB']

bars = plt.bar(categories, probabilities, color='#1f77b4')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

# Add labels and title
plt.xlabel('Road Geometry Category', fontsize=12)
plt.ylabel('Percentage of Serious/Fatal Accidents (%)', fontsize=12)
plt.title('Percentage of Serious/Fatal Accidents (Severity 1-2) by Road Geometry', fontsize=14)

# Rotate x labels if needed for better readability
plt.xticks(rotation=45, ha='right')

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('task2_4_road_geometry_serious_prob.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nStatistics and visualization saved successfully!")