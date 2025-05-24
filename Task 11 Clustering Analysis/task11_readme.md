# Task 11: Clustering Analysis of Road Accident Risk Factors

## Note on update:
The clustering has some problem as the first version wrong with the k-mean, so if you read, please just read the task11_2v2.py instead of task11_2.py - which is the version with wrong k value 

## Overview
This task analyzes road accident data to identify patterns in accident severity based on different road conditions. Using K-means clustering, we group similar combinations of light conditions, road geometry, and road surface conditions to discover which combinations present higher risks for serious or fatal accidents.

## Data Processing Workflow

### Step 1: Feature Engineering (task11_1.py)
The data processing begins by:

1. **Creating Risk Scores for Key Conditions**:
   - Each category of each condition is assigned its serious/fatal accident probability derived from previous analyses:
     - Light conditions: Using probabilities calculated in Task 1 (e.g., Dark without Lighting: 51.19%, Daylight: 37.01%)
     - Road geometry: Using probabilities calculated in Task 2 (e.g., Not at intersection: 40.52%, T intersection: 36.05%)
     - Road surface: Using probabilities calculated in Task 3 (e.g., Unpaved: 39.15%, Paved: 36.04%)
   - These probabilities are directly derived from the percentage of serious/fatal accidents in each category

2. **Applying Variance-Based Weighting**:
   - Calculate variance of serious/fatal accident probability for each condition
   - Weight each condition's influence proportionally to its variance
   - This ensures conditions with more variability in outcomes have greater influence
   - For more information, please refer to the Task 9 weighted kNN model - version 2

3. **Creating a Composite Risk Score**:
   - For each unique combination of the three conditions, we calculate SEVERE_ACCIDENT_RATE
   - This represents the actual percentage of serious/fatal accidents among all accidents that share the exact same combination of light condition, road geometry, and road surface
   - Each data point represents a specific combination, with its coordinate in 3D space determined by the weighted risk scores
   - Scale all scores to be on a 0-100 scale for easier interpretation

### Step 2: Clustering Analysis (task11_2.py)

1. **Data Visualization**:
   - Create a 3D scatter plot showing the distribution of risk factors
   - X-axis: LIGHT_RISK
   - Y-axis: GEOMETRY_RISK
   - Z-axis: SURFACE_RISK

2. **Determining Optimal Cluster Count**:
   - Use the Elbow Method to find the optimal K for K-means clustering
   - Calculate Sum of Squared Errors (SSE) for different values of K
   - Identify the "elbow point" where adding more clusters provides diminishing returns

3. **K-means Clustering**:
   - Group similar combinations of conditions into clusters
   - Use standardized features for better clustering results
   - Assign each combination to its respective cluster

### Step 3: Cluster Analysis (task11_3.py)

1. **Statistical Analysis**:
   - Calculate mean and variance of risk scores for each cluster
   - Identify characteristics that define each cluster
   - Save results to CSV for further analysis

2. **Visualization**:
   - Create formatted tables showing the top accident conditions in each cluster
   - Highlight the combinations with the highest serious/fatal accident rates

## Clustering Concepts

### K-means Clustering
K-means is an unsupervised machine learning algorithm that groups similar data points together. In this analysis:

1. **Distance Measure**: We use Euclidean distance in a 3D space of weighted risk factors
   - Points closer together in this space have similar risk profiles

2. **Variance-Based Weighting**: 
   - Features are weighted by the square root of their variance proportion
   - This ensures each feature's influence is proportional to its importance

3. **Cluster Interpretation**:
   - Each cluster represents a group of road condition combinations with similar risk characteristics
   - Higher SEVERE_ACCIDENT_RATE within a cluster indicates more dangerous combinations

## Usage

The analysis is broken into three main scripts:

1. **task11_1.py**: Processes the data and creates the risk factor dataframe
2. **task11_2.py**: Performs the clustering and generates visualizations
3. **task11_3.py**: Analyzes cluster statistics and creates table visualizations

## Output Files

- **task11_2_overview_scatter_plot.png**: 3D scatter plot of all risk factors
- **task11_2_elbow_plot.png**: Plot to determine optimal K
- **task11_2_clustered_scatter_plot.png**: 3D scatter plot showing cluster assignments
- **task11_2_cluster_0.csv/task11_2_cluster_1.csv**: Top combinations in each cluster
- **task11_3_cluster_stats.csv**: Statistical summary of each cluster
- **task11_3_cluster_0_table.png/task11_3_cluster_1_table.png**: Visualized tables of top combinations
- **task11_3_cluster_stats_table.png**: Visualized table of cluster statistics 