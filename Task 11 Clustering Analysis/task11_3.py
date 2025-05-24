'''
- This script will:
1. Calculate mean and variance of SEVERE_ACCIDENT_RATE, LIGHT_RISK, GEOMETRY_RISK, and SURFACE_RISK for each cluster (for records with COUNT >= 10)
2. Save results to a CSV file
3. Create table visualizations for each cluster with values rounded to 2 decimal places
4. Create a table visualization of the cluster statistics
5. Save tables as PNG files
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Load the cluster data
cluster0_df = pd.read_csv('Task 11 Clustering Analysis/task11_2_cluster_0.csv')
cluster1_df = pd.read_csv('Task 11 Clustering Analysis/task11_2_cluster_1.csv')

# Filter records with COUNT >= 10
cluster0_df = cluster0_df[cluster0_df['COUNT'] >= 10]
cluster1_df = cluster1_df[cluster1_df['COUNT'] >= 10]

# Calculate statistics for each cluster
stats_columns = ['SEVERE_ACCIDENT_RATE', 'LIGHT_RISK', 'GEOMETRY_RISK', 'SURFACE_RISK']
cluster_stats = []

# Calculate stats for cluster 0
cluster0_stats = {'CLUSTER': 0}
for col in stats_columns:
    cluster0_stats[f'MEAN_{col}'] = cluster0_df[col].mean()
    cluster0_stats[f'VAR_{col}'] = cluster0_df[col].var()
cluster_stats.append(cluster0_stats)

# Calculate stats for cluster 1
cluster1_stats = {'CLUSTER': 1}
for col in stats_columns:
    cluster1_stats[f'MEAN_{col}'] = cluster1_df[col].mean()
    cluster1_stats[f'VAR_{col}'] = cluster1_df[col].var()
cluster_stats.append(cluster1_stats)

# Create a DataFrame for the stats
stats_df = pd.DataFrame(cluster_stats)

# Save stats to CSV
stats_csv_path = 'Task 11 Clustering Analysis/task11_3_cluster_stats.csv'
stats_df.to_csv(stats_csv_path, index=False)
print(f"Statistics saved to {stats_csv_path}")

# Function to create a table visualization for a cluster's data rows
def create_cluster_table(df, cluster_num):
    # Round all numeric columns to 2 decimal places
    for col in df.columns:
        if col in ['LIGHT_RISK', 'GEOMETRY_RISK', 'SURFACE_RISK', 'SEVERE_ACCIDENT_RATE', 
                  'SEVERITY_MEAN', 'SEVERITY_VAR']:
            df[col] = df[col].round(2)
    
    # Select relevant columns for the table
    table_df = df[['LIGHT_GEOMETRY_SURFACE', 'LIGHT_RISK', 'GEOMETRY_RISK', 'SURFACE_RISK', 
                  'SEVERE_ACCIDENT_RATE', 'COUNT']]
    
    # Sort by SEVERE_ACCIDENT_RATE and get top rows
    table_df = table_df.sort_values('SEVERE_ACCIDENT_RATE', ascending=False).head(10)
    
    # Create a figure with minimal padding
    plt.figure(figsize=(14, len(table_df) * 0.45 + 0.8))
    ax = plt.gca()
    
    # Hide axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Create the table
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='center',
        loc='center'
    )
    
    # Set font size and adjust cell size
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Adjust column widths
    table.auto_set_column_width([0, 1, 2, 3, 4, 5])
    
    # Set title with minimal padding
    plt.title(f'Cluster {cluster_num} - Top 10 Accident Conditions with Highest Serious/Fatal Rate (COUNT >= 10)', 
              fontsize=12, pad=5)
    
    # Save the figure with very tight bounding box
    plt.savefig(f'Task 11 Clustering Analysis/task11_3_cluster_{cluster_num}_table.png', 
                bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close()
    
    print(f"Table for Cluster {cluster_num} saved as task11_3_cluster_{cluster_num}_table.png")

# Function to create a table visualization for the cluster statistics
def create_stats_table():
    # Round all values to 2 decimal places
    stats_df_rounded = stats_df.copy()
    for col in stats_df_rounded.columns:
        if col != 'CLUSTER':
            stats_df_rounded[col] = stats_df_rounded[col].round(2)
    
    # Create better column headers for display
    display_df = stats_df_rounded.copy()
    
    # Create a mapping for better column names
    col_map = {
        'CLUSTER': 'Cluster',
        'MEAN_SEVERE_ACCIDENT_RATE': 'Mean\n Significant Rate',
        'VAR_SEVERE_ACCIDENT_RATE': 'Var\nSignificant Rate',
        'MEAN_LIGHT_RISK': 'Mean\nLight Risk',
        'VAR_LIGHT_RISK': 'Var\nLight Risk',
        'MEAN_GEOMETRY_RISK': 'Mean\nGeometry Risk',
        'VAR_GEOMETRY_RISK': 'Var\nGeometry Risk',
        'MEAN_SURFACE_RISK': 'Mean\nSurface Risk',
        'VAR_SURFACE_RISK': 'Var\nSurface Risk'
    }
    
    # Reorder columns for better readability
    col_order = ['CLUSTER']
    for metric in ['SEVERE_ACCIDENT_RATE', 'LIGHT_RISK', 'GEOMETRY_RISK', 'SURFACE_RISK']:
        col_order.append(f'MEAN_{metric}')
        col_order.append(f'VAR_{metric}')
    
    display_df = display_df[col_order]
            
    # Create a figure with minimal padding
    plt.figure(figsize=(12, 2.5))
    ax = plt.gca()
    
    # Hide axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Create the table with better column labels
    table = ax.table(
        cellText=display_df.values,
        colLabels=[col_map[col] for col in display_df.columns],
        cellLoc='center',
        loc='center'
    )
    
    # Set font size and adjust cell size
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Set title with minimal padding
    plt.title('Cluster Statistics Summary (for records with COUNT >= 10)', 
              fontsize=12, pad=5)
    
    # Save the figure with very tight bounding box
    plt.savefig('Task 11 Clustering Analysis/task11_3_cluster_stats_table.png', 
                bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close()
    
    print("Cluster statistics table saved as task11_3_cluster_stats_table.png")

# Create and save table visualizations
create_cluster_table(cluster0_df, 0)
create_cluster_table(cluster1_df, 1)
create_stats_table() 