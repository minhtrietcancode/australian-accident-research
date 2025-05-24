'''
- This task will process the data for my clustering analysis and then return a dataframe of 
  the processed data to be used in the next function of the clustering analysis.

- Function signature:
    + def task11_1():
        - This function will process the data for my clustering analysis and then return a dataframe of 
          the processed data to be used in the next function of the clustering analysis.

- How to preprocess the data:
    + Create representative features for each condition (light, geometry, road surface):
      - Map each category of each condition to its serious/fatal accident probability from Tasks 1-3
      - Scale these probabilities to be over 100 (e.g., 0.55 becomes 55)
      - Apply the variance-based weighting from Task 9 (version 2) to give appropriate influence to each condition
      
    + Create a composite SEVERE_ACCIDENT_RATE column:
      - For each unique combination of the three conditions, calculate the probability of serious/fatal accidents
      - Scale this to be over 100 as well
      
    + The resulting dataframe will have these key columns:
      - LIGHT_RISK: Weighted risk score for light condition (scaled 0-100)
      - GEOMETRY_RISK: Weighted risk score for road geometry (scaled 0-100)
      - SURFACE_RISK: Weighted risk score for road surface (scaled 0-100)
      - LIGHT_GEOMETRY_SURFACE: Combined column showing the original categories
      - SEVERE_ACCIDENT_RATE: Overall risk score for each unique combination (scaled 0-100)
      
- The visualization and clustering plan:
    + Step 1: Create a 3D scatter plot visualization
      - Each point represents a unique combination of the three conditions
      - Use a color gradient from light to dark based on the SEVERE_ACCIDENT_RATE
      - This provides an overview of the risk distribution in the feature space
      
    + Step 2: Determine optimal k for k-means clustering
      - Run k-means with different values of k
      - Plot Sum of Squared Errors against k to find the elbow point
      - Select the best k based on this analysis
      
    + Step 3: Perform k-means clustering with the optimal k
      - Apply the algorithm to group similar combinations together
      - Create a new 3D scatter plot with colors representing cluster membership
      - Analyze the characteristics of each cluster to identify patterns
      
    + The 3D feature space will use:
      - X-axis: LIGHT_RISK
      - Y-axis: GEOMETRY_RISK
      - Z-axis: SURFACE_RISK
'''
import pandas as pd

'''
- Function signature:
    + def task11_1():
        - This function will process the data for my clustering analysis and then return a dataframe of 
          the processed data to be used in the next function of the clustering analysis.
'''
def task11_1():
  # read the csv file of serious/fatal accidents percentage by all conditions
  light_condition_prob = pd.read_csv('Task 1 LIGHT_CONDITION VS SEVERITY/task1_4_light_condition_serious_prob.csv')
  road_geometry_prob = pd.read_csv('Task 2 ROAD_GEOMETRY vs SEVERITY/task2_4_road_geometry_serious_prob.csv')
  road_surface_prob = pd.read_csv('Task 3 ROAD_SURFACE vs SEVERITY/task3_4_road_surface_serious_prob.csv')

  # Now calculate the variance of the percentage of serious/fatal accidents for each condition
  # modify to just calculate the variance to exlcude the value of "Unknown"
  light_condition_variance = light_condition_prob[light_condition_prob['LIGHT_CONDITION_CATEGORY'] != 'Unknown']['SERIOUS_AND_FATAL_ACCIDENT_PROB'].var()
  road_geometry_variance = road_geometry_prob[road_geometry_prob['ROAD_GEOMETRY_CATEGORY'] != 'Unknown']['SERIOUS_AND_FATAL_ACCIDENT_PROB'].var()
  road_surface_variance = road_surface_prob[road_surface_prob['ROAD_SURFACE_CATEGORY'] != 'Unknown']['SERIOUS_AND_FATAL_ACCIDENT_PROB'].var()

  # Now lets calculate the weight for each condition based on the variance, with the formula:
  # weight = variance / sum of variance
  variance_sum = light_condition_variance + road_geometry_variance + road_surface_variance
  light_condition_weight = light_condition_variance / variance_sum
  road_geometry_weight = road_geometry_variance / variance_sum
  road_surface_weight = road_surface_variance / variance_sum

  # Read the datasets
  accident = pd.read_csv('Data Preprocess/updated_accident.csv')
  vehicle = pd.read_csv('Data Preprocess/updated_vehicle.csv')

  # Merge the datasets
  merged_df = pd.merge(accident, vehicle, on='ACCIDENT_NO', how='inner')

  # Define the mapping dictionaries just like in task9_1.py
  light_condition_mapping = {
      'Dark without Lighting': 51.19,
      'Dark with Lighting': 41.62,
      'Daylight': 37.01,
      'Limited Light': 35.19,
      'Unknown': 19.36
  }

  road_geometry_mapping = {
      'Not at intersection': 40.52,
      'Dead end': 40.12,
      'Unknown': 36.53,
      'Multiple intersection': 36.09,
      'T intersection': 36.05,
      'Y intersection': 34.75,
      'Road closure': 33.33,
      'Cross intersection': 33.01,
      'Private property': 27.27
  }

  road_surface_mapping = {
      'Unpaved': 39.15,
      'Gravel': 38.51,
      'Paved': 36.04,
      'Unknown': 17.89
  }

  # Create the risk columns
  merged_df['SIGNIFICANT_SEVERITY_RISK_LIGHT_CONDITION'] = merged_df['CATEGORIZED_LIGHT_CONDITION'].map(light_condition_mapping)
  merged_df['SIGNIFICANT_SEVERITY_RISK_ROAD_GEOMETRY'] = merged_df['ROAD_GEOMETRY_DESC'].map(road_geometry_mapping)
  merged_df['SIGNIFICANT_SEVERITY_RISK_ROAD_SURFACE'] = merged_df['CATEGORIZED_ROAD_SURFACE'].map(road_surface_mapping)

  # Now create the weighted columns, remember to take the square root of the weight as when we calculate the distance,
  # the difference between each variable will be squared, so we need to take the square root to make the difference actually 
  # reflected the weight of each condition
  # recall the Euclidean distance: d = sqrt(sum((x1 - x2)^2) + sum((y1 - y2)^2) + sum((z1 - z2)^2) + ...) 
  merged_df['LIGHT_RISK'] = merged_df['SIGNIFICANT_SEVERITY_RISK_LIGHT_CONDITION'] * (light_condition_weight ** 0.5)
  merged_df['GEOMETRY_RISK'] = merged_df['SIGNIFICANT_SEVERITY_RISK_ROAD_GEOMETRY'] * (road_geometry_weight ** 0.5) 
  merged_df['SURFACE_RISK'] = merged_df['SIGNIFICANT_SEVERITY_RISK_ROAD_SURFACE'] * (road_surface_weight ** 0.5)

  # Create a combined column for the three conditions
  merged_df['LIGHT_GEOMETRY_SURFACE'] = (
      merged_df['CATEGORIZED_LIGHT_CONDITION'] + ' | ' + 
      merged_df['ROAD_GEOMETRY_DESC'] + ' | ' + 
      merged_df['CATEGORIZED_ROAD_SURFACE']
  )

  # Create indicator for serious/fatal accidents (SEVERITY values 1 and 2 are serious/fatal)
  merged_df['IS_SERIOUS_FATAL'] = merged_df['SEVERITY'].isin([1, 2])

  # Group by the three risk columns and the combined condition column
  grouped_df = merged_df.groupby([
      'LIGHT_RISK', 'GEOMETRY_RISK', 'SURFACE_RISK', 'LIGHT_GEOMETRY_SURFACE'
  ]).agg({
      'IS_SERIOUS_FATAL': 'mean',  # Percentage of serious/fatal accidents
      'ACCIDENT_NO': 'count',      # Count of accidents in this group
      'SEVERITY': ['mean', 'var']  # Mean and variance of severity
  }).reset_index()

  # Rename the columns
  grouped_df.columns = [
      'LIGHT_RISK', 'GEOMETRY_RISK', 'SURFACE_RISK', 'LIGHT_GEOMETRY_SURFACE',
      'SEVERE_ACCIDENT_RATE', 'COUNT', 'SEVERITY_MEAN', 'SEVERITY_VAR'
  ]
  
  # Scale the SEVERE_ACCIDENT_RATE to be over 100
  grouped_df['SEVERE_ACCIDENT_RATE'] = grouped_df['SEVERE_ACCIDENT_RATE'] * 100
  
  # Select the required columns
  result_df = grouped_df[[
      'LIGHT_RISK', 'GEOMETRY_RISK', 'SURFACE_RISK', 
      'LIGHT_GEOMETRY_SURFACE', 'SEVERE_ACCIDENT_RATE', 
      'SEVERITY_MEAN', 'SEVERITY_VAR', 'COUNT'
  ]]
  
  return result_df

