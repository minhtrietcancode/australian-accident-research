'''
    - Things we have to do for this task
        + Generate pie chart showing distribution of light conditions across all accidents
           → Save as: task1_2_light_condition_pie.png
        + Create stacked bar chart showing distribution of light conditions for each SEVERITY level
           → Save as: task1_2_light_condition_severity_stacked.png
        + Create pie chart showing percentage of serious/fatal accidents (SEVERITY 1-2) by light condition
           → Save as: task1_2_light_condition_severity_proportion.png
           (it means the whole of the pie be the total number of severity 1 and 2, and then each part of the pie will
           be the contribution of each light condition to the pie)

    - Dataset to use: updated_accident.csv because we just do some overview between LIGHT_CONDITION and SEVERITY here
      no need for merging or using any other dataset here, use the CATEGORIZED_LIGHT_CONDITION columns here
'''

'''
Recall about the columns once again here 
    - SEVERITY: value from 1 to 4: with this definition 
            1 Fatal accident 
            2 Serious injury accident 
            3 Other injury accident 
            4 Non injury accident
    
    - CATEGORIZED_LIGHT_CONDITION:
            "Daylight"
            "Limited Light"
            "Dark with Lighting"
            "Dark without Lighting"
            "Unknown"
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
accident = pd.read_csv('Data Preprocess/updated_accident.csv')

# ===== 1. Pie Chart for Light Condition Distribution =====
plt.figure(figsize=(10, 6))
# Count the occurrences of each light condition
light_counts = accident['CATEGORIZED_LIGHT_CONDITION'].value_counts()

# Define better colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create pie chart with no labels on the pie itself, but with a legend instead
plt.pie(light_counts, autopct='%1.1f%%', startangle=90, shadow=False, colors=colors)
plt.title('Distribution of Light Conditions Across All Accidents', fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Add legend
plt.legend(light_counts.index, title="Light Conditions", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('Task 1 LIGHT_CONDITION VS SEVERITY/task1_2_light_condition_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# ===== 2. Stacked Bar Chart for Light Condition by Severity =====
plt.figure(figsize=(12, 7))

# Create a cross-tabulation of severity and light condition
severity_light = pd.crosstab(accident['SEVERITY'], accident['CATEGORIZED_LIGHT_CONDITION'])

# Convert to percentage within each severity level
severity_light_pct = severity_light.div(severity_light.sum(axis=1), axis=0) * 100

# Create stacked bar chart
severity_light_pct.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 7))
plt.title('Distribution of Light Conditions by Accident Severity Level', fontsize=14)
plt.xlabel('Severity Level (1: Fatal, 2: Serious, 3: Other Injury, 4: Non-Injury)', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.legend(title='Light Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Task 1 LIGHT_CONDITION VS SEVERITY/task1_2_light_condition_severity_stacked.png', dpi=300, bbox_inches='tight')
plt.close()

# ===== 3. Pie Chart for Serious/Fatal Accidents by Light Condition =====
plt.figure(figsize=(10, 6))

# Filter for serious and fatal accidents (SEVERITY 1-2)
serious_fatal = accident[accident['SEVERITY'].isin([1, 2])]

# Count the occurrences of each light condition in serious/fatal accidents
serious_fatal_light = serious_fatal['CATEGORIZED_LIGHT_CONDITION'].value_counts()

# Define better colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create pie chart with no labels on the pie itself, but with a legend instead
plt.pie(serious_fatal_light, autopct='%1.1f%%', startangle=90, shadow=False, colors=colors)
plt.title('Distribution of Light Conditions for Serious/Fatal Accidents (Severity 1-2)', fontsize=14)
plt.axis('equal')

# Add legend
plt.legend(serious_fatal_light.index, title="Light Conditions", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('Task 1 LIGHT_CONDITION VS SEVERITY/task1_2_light_condition_severity_proportion.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some summary statistics to verify our analysis
print("\nDistribution of Light Conditions Across All Accidents:")
print(light_counts)

print("\nCross-tabulation of Severity and Light Condition:")
print(severity_light)

print("\nDistribution of Light Conditions for Serious/Fatal Accidents (Severity 1-2):")
print(serious_fatal_light)
