import pandas as pd
import numpy as np

'''
    - Categorize the ROAD_SURFACE_TYPE_DESC to generate:
        + CATEGORIZED_ROAD_SURFACE
        + DECODED_ROAD_SURFACE
'''
# Load the csv file to the dataframe
vehicle = pd.read_csv('original-datasets/vehicle.csv')

# Extract the targeted columns
surface = vehicle['ROAD_SURFACE_TYPE_DESC']

# Here is the value_counts for this column
# ROAD_SURFACE_TYPE_DESC
# Paved        309360
# Gravel        11859
# Not known      2733
# Unpaved        1936
# NaN               1
# Name: count, dtype: int64

# Remove rows where ROAD_SURFACE_TYPE_DESC is NaN as we just have 1 NaN value - not a significant number of NaN value
# at all
vehicle = vehicle.dropna(subset=['ROAD_SURFACE_TYPE_DESC'])

# Create CATEGORIZED_ROAD_SURFACE column
category_mapping_road_surface = {
    "Paved": "Paved",
    "Gravel": "Gravel",
    "Unpaved": "Unpaved",
    "Not known": "Unknown"  # Map "Not known" to "Unknown"
}

# Apply the mapping to create CATEGORIZED_ROAD_SURFACE
vehicle['CATEGORIZED_ROAD_SURFACE'] = vehicle['ROAD_SURFACE_TYPE_DESC'].map(category_mapping_road_surface)

# Create DECODED_ROAD_SURFACE column
decoded_mapping_road_surface = {
    "Paved": 3,    # best condition
    "Gravel": 2,   # moderate condition
    "Unpaved": 1,  # worst condition
    "Not known": np.nan  # Map "Not known" to NaN to exclude from correlation analysis
}

# Apply the mapping to create DECODED_ROAD_SURFACE
vehicle['DECODED_ROAD_SURFACE'] = vehicle['ROAD_SURFACE_TYPE_DESC'].map(decoded_mapping_road_surface)

# Select only ACCIDENT_NO and the columns related to ROAD_SURFACE_TYPE_DESC
selected_columns = [
    'ACCIDENT_NO',

    'ROAD_SURFACE_TYPE', # encoded column for the road surface type desc in case we need
    'ROAD_SURFACE_TYPE_DESC',
    'CATEGORIZED_ROAD_SURFACE',
    # 'DECODED_ROAD_SURFACE' can uncomment if would like to include in the future,
    # however currently i remove that as i think this column has some issue
]

# Create a new dataframe with only the selected columns
updated_vehicle = vehicle[selected_columns]

# Save the selected columns to a new CSV file
updated_vehicle.to_csv('Data Preprocess/updated_vehicle.csv', index=False)
