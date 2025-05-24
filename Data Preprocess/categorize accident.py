import pandas as pd
import numpy as np

'''
    - Handle the LIGHT_CONDITION to generate:  
        + CATEGORIZED_LIGHT_CONDITION 
        + DECODED_LIGHT_CONDITION 
'''
# Read the dataset into the dataframe
accident = pd.read_csv('original-datasets/accident.csv')

# Note that i has verified before that this column does not have NaN value
# Here is the value count for this column
# LIGHT_CONDITION
# 1    119343
# 3     27332
# 2     15407
# 5      9449
# 9      4772
# 6      2005
# 4       387
# Name: count, dtype: int64
light = accident['LIGHT_CONDITION']

# Remember that this is just the meaning of each decoded value of the LIGHT_CONDITION column,
# just put a dictionary here to easily see everything
light_condition_meaning = {
    1: "Day",
    2: "Dusk/Dawn",
    3: "Dark street lights on",
    4: "Dark street lights off",
    5: "Dark no street lights",
    6: "Dark street lights unknown",
    9: "Unknown"
}

# Here is the mapping for CATEGORIZED_LIGHT_CONDITION column
# With the 6: Dark street lights unknown, i choose to make it to be Unknown
# as its value count is 2005 - just about 1.1% of the dataset, not that important, make
# it to be Unknown to avoid making mistake for the analysis
category_mapping_light_condition = {
    1: "Daylight",
    2: "Limited Light",
    3: "Dark with Lighting",
    4: "Dark without Lighting",
    5: "Dark without Lighting",
    6: "Unknown",
    9: "Unknown"
}
accident['CATEGORIZED_LIGHT_CONDITION'] = accident['LIGHT_CONDITION'].map(category_mapping_light_condition)

# Here is the mapping for DECODED_LIGHT_CONDITION
decoded_mapping_light_condition = {
    1: 4,  # Day (best conditions)
    2: 3,  # Dusk/dawn
    3: 2,  # Dark street lights on
    4: 1,  # Dark street lights off (worst conditions)
    5: 1,  # Dark no street lights (worst conditions)
    6: np.nan,  # Dark street lights unknown (exclude)
    9: np.nan   # Unknown (exclude)
}
accident['DECODED_LIGHT_CONDITION'] = accident['LIGHT_CONDITION'].map(decoded_mapping_light_condition)

'''
    - Handle the ROAD_GEOMETRY_DESC to generate
        + CATEGORIZED_ROAD_GEOMETRY
        + DECODED_ROAD_GEOMETRY
'''
# Extract the column ROAD_GEOMETRY_DESC
geometry = accident['ROAD_GEOMETRY_DESC']

# Here is the value counts for this column, no NaN values at all 
# ROAD_GEOMETRY_DESC
# Not at intersection      92444
# Cross intersection       41039
# T intersection           40656
# Multiple intersection     3486
# Y intersection             610
# Unknown                    271
# Dead end                   172
# Private property            11
# Road closure                 6
# Name: count, dtype: int64

# Mapping for the column CATEGORIZED_ROAD_GEOMETRY
category_mapping_road_geometry = {
    "Not at intersection": "Not at intersection",
    "Cross intersection": "Intersection",
    "T intersection": "Intersection",
    "Multiple intersection": "Intersection",
    "Y intersection": "Intersection",
    "Dead end": "Special Road Feature",
    "Road closure": "Special Road Feature",
    "Unknown": "Other/Unknown",
    "Private property": "Other/Unknown"
}
accident['CATEGORIZED_ROAD_GEOMETRY'] = accident['ROAD_GEOMETRY_DESC'].map(category_mapping_road_geometry)

# Mapping for the column DECODED_ROAD_GEOMETRY
decoded_mapping_road_geometry = {
    "Not at intersection": 5,     # simplest road geometry
    "Cross intersection": 2,
    "T intersection": 3,
    "Multiple intersection": 1,   # most complex/dangerous intersections
    "Y intersection": 1,          # most complex/dangerous intersections
    "Dead end": 4,
    "Road closure": 4,
    "Unknown": np.nan,            # exclude from correlation analysis
    "Private property": np.nan    # exclude from correlation analysis
}
accident['DECODED_ROAD_GEOMETRY'] = accident['ROAD_GEOMETRY_DESC'].map(decoded_mapping_road_geometry)

# Select only the columns related to ROAD_GEOMETRY_DESC, LIGHT_CONDITION, and SEVERITY, ACCIDENT_NO
selected_columns = [
    # Obviously the ACCIDENT_NO
    'ACCIDENT_NO',

    # Light condition related columns
    'LIGHT_CONDITION',
    'CATEGORIZED_LIGHT_CONDITION',
    # 'DECODED_LIGHT_CONDITION', uncomment if we want to include this in the future, as
    # I think this column has some issues, can consider later

    # Road geometry related columns
    'ROAD_GEOMETRY', # encoded columns of ROAD_GEOMETRY_DESC in case we need
    'ROAD_GEOMETRY_DESC',
    'CATEGORIZED_ROAD_GEOMETRY',
    # 'DECODED_ROAD_GEOMETRY', uncomment if we want to include this in the future, as
    # I think this column has some issues, can consider later

    # Severity columns
    'SEVERITY'
]

# Create a new dataframe with only the selected columns
updated_accident = accident[selected_columns]

# And then save as the new CSV file
updated_accident.to_csv('Data Preprocess/updated_accident.csv', index=False)

