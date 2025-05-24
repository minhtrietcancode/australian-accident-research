'''
    - Task to do:
        + Generate pie chart showing distribution of road surfaces
          → Save as: task3_2_road_surface_pie.png
        + Create stacked bar chart showing distribution of road surfaces for each SEVERITY level
          → Save as: task3_2_road_surface_severity_stacked.png
        + Create pie chart showing percentage of serious/fatal accidents by road surface
          → Save as: task3_2_road_surface_severity_proportion.png
          (it means the whole of the pie be the total number of severity 1 and 2, and then each part of the pie
          will be the contribution of each light condition to the pie)

    - Datasets to use: the merging of update_accident.csv and update_vehicle.csv (on = ACCIDENT_NO) as SEVERITY column
      is in updated_accident.csv and CATEGORIZED_ROAD_SURFACE is in updated_vehicle.csv
'''

'''
NOTE ON COLUMNS TO USE 
    - SEVERITY (update_accident.csv): value from 1 to 4: with this definition 
            1 Fatal accident 
            2 Serious injury accident 
            3 Other injury accident 
            4 Non injury accident
    
    - CATEGORIZED_ROAD_SURFACE (updated_vehicle.csv):
            "Paved"
            "Gravel"
            "Unpaved"
            "Unknown"
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the datasets
accident = pd.read_csv('../Data Preprocess/updated_accident.csv')
vehicle = pd.read_csv('../Data Preprocess/updated_vehicle.csv')

# Merge datasets on ACCIDENT_NO
merged_data = pd.merge(accident, vehicle, on='ACCIDENT_NO')

# ===== 1. Pie Chart for Road Surface Distribution =====
plt.figure(figsize=(10, 6))
# Count the occurrences of each road surface
road_surface_counts = merged_data['CATEGORIZED_ROAD_SURFACE'].value_counts()

# Define better colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create pie chart with no labels on the pie itself, but with a legend instead
plt.pie(road_surface_counts, autopct='%1.1f%%', startangle=90, shadow=False, colors=colors)
plt.title('Distribution of Road Surfaces Across All Accidents', fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Add legend
plt.legend(road_surface_counts.index, title="Road Surfaces", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('task3_2_road_surface_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# ===== 2. Stacked Bar Chart for Road Surface by Severity =====
plt.figure(figsize=(12, 7))

# Create a cross-tabulation of severity and road surface
severity_road = pd.crosstab(merged_data['SEVERITY'], merged_data['CATEGORIZED_ROAD_SURFACE'])

# Convert to percentage within each severity level
severity_road_pct = severity_road.div(severity_road.sum(axis=1), axis=0) * 100

# Create stacked bar chart
severity_road_pct.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 7))
plt.title('Distribution of Road Surfaces by Accident Severity Level', fontsize=14)
plt.xlabel('Severity Level (1: Fatal, 2: Serious, 3: Other Injury, 4: Non-Injury)', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.legend(title='Road Surface', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('task3_2_road_surface_severity_stacked.png', dpi=300, bbox_inches='tight')
plt.close()

# ===== 3. Pie Chart for Serious/Fatal Accidents by Road Surface =====
plt.figure(figsize=(10, 6))

# Filter for serious and fatal accidents (SEVERITY 1-2)
serious_fatal = merged_data[merged_data['SEVERITY'].isin([1, 2])]

# Count the occurrences of each road surface in serious/fatal accidents
serious_fatal_road = serious_fatal['CATEGORIZED_ROAD_SURFACE'].value_counts()

# Define better colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create pie chart with no labels on the pie itself, but with a legend instead
plt.pie(serious_fatal_road, autopct='%1.1f%%', startangle=90, shadow=False, colors=colors)
plt.title('Distribution of Road Surfaces for Serious/Fatal Accidents (Severity 1-2)', fontsize=14)
plt.axis('equal')

# Add legend
plt.legend(serious_fatal_road.index, title="Road Surfaces", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('task3_2_road_surface_severity_proportion.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some summary statistics to verify our analysis
print("\nDistribution of Road Surfaces Across All Accidents:")
print(road_surface_counts)

print("\nCross-tabulation of Severity and Road Surface:")
print(severity_road)

print("\nDistribution of Road Surfaces for Serious/Fatal Accidents (Severity 1-2):")
print(serious_fatal_road)