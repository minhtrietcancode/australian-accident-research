'''
    - Task to do;
        + Generate pie chart showing distribution of road geometries
          → Save as: task2_2_road_geometry_pie.png
        + Create stacked bar chart showing distribution of road geometries for each SEVERITY level
          → Save as: task2_2_road_geometry_severity_stacked.png
        + Create pie chart showing percentage of serious/fatal accidents by road geometry
          → Save as: task2_2_road_geometry_severity_proportion.png
          (it means the whole of the pie be the total number of severity 1 and 2, and then each part of the pie
          will be the contribution of each light condition to the pie)

    - Datasets to use:
        + updated_accident.csv
'''

'''
NOTE ON THE COLUMNS TO USE 
    - SEVERITY: value from 1 to 4: with this definition 
            1 Fatal accident 
            2 Serious injury accident 
            3 Other injury accident 
            4 Non injury accident
    
    - CATEGORIZED_ROAD_GEOMETRY
            "Not at intersection"
            "Intersection"
            "Special Road Feature"
            "Other/Unknown"
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
accident = pd.read_csv('Data Preprocess/updated_accident.csv')

# ===== 1. Pie Chart for Road Geometry Distribution =====
plt.figure(figsize=(10, 6))
# Count the occurrences of each road geometry
geometry_counts = accident['CATEGORIZED_ROAD_GEOMETRY'].value_counts()

# Define better colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Create pie chart with no labels on the pie itself, but with a legend instead
plt.pie(geometry_counts, autopct='%1.1f%%', startangle=90, shadow=False, colors=colors)
plt.title('Distribution of Road Geometries Across All Accidents', fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Add legend
plt.legend(geometry_counts.index, title="Road Geometries", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('Task 2 ROAD_GEOMETRY vs SEVERITY/task2_2_road_geometry_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# ===== 2. Stacked Bar Chart for Road Geometry by Severity =====
plt.figure(figsize=(12, 7))

# Create a cross-tabulation of severity and road geometry
severity_geometry = pd.crosstab(accident['SEVERITY'], accident['CATEGORIZED_ROAD_GEOMETRY'])

# Convert to percentage within each severity level
severity_geometry_pct = severity_geometry.div(severity_geometry.sum(axis=1), axis=0) * 100

# Create stacked bar chart
severity_geometry_pct.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 7))
plt.title('Distribution of Road Geometries by Accident Severity Level', fontsize=14)
plt.xlabel('Severity Level (1: Fatal, 2: Serious, 3: Other Injury, 4: Non-Injury)', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.legend(title='Road Geometry', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Task 2 ROAD_GEOMETRY vs SEVERITY/task2_2_road_geometry_severity_stacked.png', dpi=300, bbox_inches='tight')
plt.close()

# ===== 3. Pie Chart for Serious/Fatal Accidents by Road Geometry =====
plt.figure(figsize=(10, 6))

# Filter for serious and fatal accidents (SEVERITY 1-2)
serious_fatal = accident[accident['SEVERITY'].isin([1, 2])]

# Count the occurrences of each road geometry in serious/fatal accidents
serious_fatal_geometry = serious_fatal['CATEGORIZED_ROAD_GEOMETRY'].value_counts()

# Define better colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Create pie chart with no labels on the pie itself, but with a legend instead
plt.pie(serious_fatal_geometry, autopct='%1.1f%%', startangle=90, shadow=False, colors=colors)
plt.title('Distribution of Road Geometries for Serious/Fatal Accidents (Severity 1-2)', fontsize=14)
plt.axis('equal')

# Add legend
plt.legend(serious_fatal_geometry.index, title="Road Geometries", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('Task 2 ROAD_GEOMETRY vs SEVERITY/task2_2_road_geometry_severity_proportion.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some summary statistics to verify our analysis
print("\nDistribution of Road Geometries Across All Accidents:")
print(geometry_counts)

print("\nCross-tabulation of Severity and Road Geometry:")
print(severity_geometry)

print("\nDistribution of Road Geometries for Serious/Fatal Accidents (Severity 1-2):")
print(serious_fatal_geometry)
