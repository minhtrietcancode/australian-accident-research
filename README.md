# Road Accident Severity Analysis

Note that this is developed as part of Assignment 2 for COMP20008 - Element of Data Processing at University of Melbourne Semester 1, 2025. The code is developed fully by myself, meanwhile the report is written by our group. 

For detailed analysis, please refer to the [report](./Victoria%20Road%20Crash%20Research.pdf). This README file will only provide a brief overview of the project, and which file is responsible for which part of the analysis.

## Research Question
**How do external factors influence accident severity?**

This research focuses on analyzing how three main external factors affect the severity of road accidents:
- Light conditions
- Road geometry
- Road surface type

## Dataset Overview

The analysis is based on two primary datasets:

### 1. accident.csv
- Contains 178,695 records of road accidents
- Key columns:
  - `ACCIDENT_NO`: Unique identifier for each accident
  - `LIGHT_CONDITION`: Lighting condition at the time of the accident
  - `ROAD_GEOMETRY` and `ROAD_GEOMETRY_DESC`: Road configuration information
  - `SEVERITY`: Severity level of the accident
  - Additional information like date, time, number of vehicles, and casualties

### 2. vehicle.csv
- Contains 325,889 records of vehicles involved in accidents
- Key columns:
  - `ACCIDENT_NO`: Links to the accident dataset
  - `ROAD_SURFACE_TYPE` and `ROAD_SURFACE_TYPE_DESC`: Information about the road surface
  - Additional vehicle details like make, model, year, and damage

## Data Preprocessing

The preprocessing phase involved several key steps to prepare the data for analysis:

### Light Condition Processing
- Original `LIGHT_CONDITION` values (1-9) were categorized into:
  - "Daylight" (code 1)
  - "Limited Light" (code 2 - Dusk/Dawn)
  - "Dark with Lighting" (code 3 - Dark street lights on)
  - "Dark without Lighting" (codes 4, 5 - Dark street lights off/no street lights)
  - "Unknown" (codes 6, 9 - Dark street lights unknown/Unknown)

### Road Geometry Processing
- Original `ROAD_GEOMETRY_DESC` values were categorized into:
  - "Not at intersection"
  - "Intersection" (Cross, T, Multiple, Y intersections)
  - "Special Road Feature" (Dead end, Road closure)
  - "Other/Unknown" (Unknown, Private property)

### Road Surface Processing
- Original `ROAD_SURFACE_TYPE_DESC` values were categorized into:
  - "Paved"
  - "Gravel"
  - "Unpaved"
  - "Unknown" (Not known)

The preprocessing created two updated datasets:
- `updated_accident.csv`: Contains original and categorized light condition and road geometry
- `updated_vehicle.csv`: Contains original and categorized road surface type

## Analysis Structure

The analysis is structured in layers of increasing complexity:

### Layer 1: Univariate Analysis (Tasks 1-3)
Examining how each individual factor relates to accident severity.

#### Task 1: Light Condition vs. Severity
- Overview analysis of the relationship between light conditions and accident severity
- Proportional analysis of light conditions across severity levels
- Statistical analysis including mean and variance of severity within each light condition category
- Visualizations including pie charts, stacked bar charts, and comparative bar charts

#### Task 2: Road Geometry vs. Severity
- Overview analysis of the relationship between road geometry and accident severity
- Proportional analysis of road geometry across severity levels
- Statistical analysis including mean and variance of severity within each road geometry category
- Visualizations including pie charts, stacked bar charts, and comparative bar charts

#### Task 3: Road Surface vs. Severity
- Overview analysis of the relationship between road surface types and accident severity
- Proportional analysis of road surface types across severity levels
- Statistical analysis including mean and variance of severity within each road surface category
- Visualizations including pie charts, stacked bar charts, and comparative bar charts

#### Task 4: Mutual Information Correlation Analysis
- Calculated mutual information (MI) between:
  - Each individual factor (Light Condition, Road Geometry, Road Surface) and Severity
  - Pairwise combinations of factors (Light+Geometry, Light+Surface, Geometry+Surface) and Severity
  - Full combination of all three factors and Severity
- Used original encoded columns to avoid bias from categorization
- Visualized normalized mutual information scores to compare the strength of relationships

### Layer 2: Bivariate Analysis (Tasks 5-7)
Examining how pairs of factors together relate to accident severity.

#### Task 5: Light Condition + Road Geometry vs. Severity
- Statistical analysis of the combined effect of light conditions and road geometry on accident severity
- Proportional analysis across different combinations of categories
- Visualizations of accident severity distributions for combined factor categories

#### Task 6: Light Condition + Road Surface vs. Severity
- Statistical analysis of the combined effect of light conditions and road surface on accident severity
- Proportional analysis across different combinations of categories
- Visualizations of accident severity distributions for combined factor categories

#### Task 7: Road Geometry + Road Surface vs. Severity
- Statistical analysis of the combined effect of road geometry and road surface on accident severity
- Proportional analysis across different combinations of categories
- Visualizations of accident severity distributions for combined factor categories

### Layer 3: Multivariate Analysis (Task 8)
Examining how all three factors together relate to accident severity.

#### Task 8: Full Combination vs. Severity
- Statistical analysis of the combined effect of all three factors on accident severity
- Proportional analysis across different combinations of categories
- Identified the distribution of severity across various combinations of light conditions, road geometry, and road surface

### Supervised Learning Models (Tasks 9-10)
Using machine learning to predict accident severity based on the external factors.

#### Task 9: k-NN Classification
- Implemented k-Nearest Neighbors models to predict accident severity
- Created two versions:
  - Version 1: Using risk scores derived from probability of serious/fatal accidents
  - Version 2: Incorporating variance-based weighted features to prioritize more important factors
- Used 5-fold cross-validation with k=200

#### Task 10: Decision Tree Classification
- Implemented Decision Tree models to predict accident severity
- Created an interpretable model that can show the decision rules for classification
- Visualized the decision tree to understand which factors and thresholds were most important

### Unsupervised Learning (Task 11)
Using clustering to discover patterns in the data.

#### Task 11: Clustering Analysis
- Performed K-means clustering to group similar combinations of road conditions
- Used variance-based weighting to ensure each factor's influence was proportional to its importance
- Determined optimal number of clusters using the Elbow Method
- Analyzed cluster characteristics to identify patterns of high-risk combinations

## Conclusion

This research follows a structured analytical approach to understand the relationship between external factors and accident severity. The analysis progresses through multiple layers:

1. Starting with detailed univariate analysis of individual factors (Tasks 1-3), establishing a baseline understanding of how each factor correlates with accident severity
2. Advancing to mutual information analysis (Task 4) to quantify the strength of relationships between factors and severity
3. Exploring bivariate relationships (Tasks 5-7) to understand how pairs of factors interact to influence severity
4. Investigating multivariate relationships (Task 8) to examine the combined effect of all three factors
5. Applying supervised learning models (Tasks 9-10) to verify insights gained from previous analyses and establish a foundation for predictive modeling
6. Using unsupervised learning (Task 11) to discover natural patterns in the data that might not be apparent through direct analysis

This progressive analytical approach provides a comprehensive understanding of how external factors influence accident severity, creating a solid foundation for future research and potential real-world applications in road safety planning and infrastructure development. 