'''
- In this version, I will add the order of the conditions into the model and create a weight for each condition
- The order of the conditions is: Light condition --> Road geometry --> Road surface
- And now I will illustrate the way to create the weight for each condition, this can be done
  through the following steps:
  + Calculate the variance of the percentage of serious/fatal accidents for each condition
  + Calculate the weight for each condition based on the variance, with the formula:
    - weight = variance / sum of variance
  + Now we have the weight for each condition, we can use this to create a new column in the dataset
  + The reason for doing this is because the k-NN model is not able to capture the order of the conditions,
    so we need to add the order of the conditions into the model and create a weight for each condition. And we 
    calculate the weight based on the variance because the variance is a good indicator of the spread of the data,
    and the more spread the data is, the more important the condition is. With the physical meaning, big variance 
    --> big difference of percentage when the category of a condition is changed --> that means the condition is
    more sensitive to the severity of the accident --> so we should assign a higher weight to this condition

    - Now here is the steps to implement the updated k-NN model:
      + Calculate the weight for each condition based on the variance
      + Create the columns as task9_1.py
      + Mutiply each columns with the corresponding weight
      + Use the new columns to train the k-NN model
      + Test the model and evaluate the performance
    
    - Remember we still use k = 200, and conduct 5-fold cross-validation
'''

import pandas as pd
import math

# read the csv file of serious/fatal accidents percentage by all conditions
light_condition_prob = pd.read_csv('Task 1 LIGHT_CONDITION VS SEVERITY/task1_4_light_condition_serious_prob.csv')
road_geometry_prob = pd.read_csv('Task 2 ROAD_GEOMETRY vs SEVERITY/task2_4_road_geometry_serious_prob.csv')
road_surface_prob = pd.read_csv('Task 3 ROAD_SURFACE vs SEVERITY/task3_4_road_surface_serious_prob.csv')

# Now calculate the variance of the percentage of serious/fatal accidents for each condition
# modify to just calculate the variance to exlcude the value of "Unknown"
light_condition_variance = light_condition_prob[light_condition_prob['LIGHT_CONDITION_CATEGORY'] != 'Unknown']['SERIOUS_AND_FATAL_ACCIDENT_PROB'].var()
road_geometry_variance = road_geometry_prob[road_geometry_prob['ROAD_GEOMETRY_CATEGORY'] != 'Unknown']['SERIOUS_AND_FATAL_ACCIDENT_PROB'].var()
road_surface_variance = road_surface_prob[road_surface_prob['ROAD_SURFACE_CATEGORY'] != 'Unknown']['SERIOUS_AND_FATAL_ACCIDENT_PROB'].var()

'''
And here is the result of the variance:
51.215238961779825
17.896820881999016
2.7086596545344506
'''

# Now lets calculate the weight for each condition based on the variance, with the formula:
# weight = variance / sum of variance
variance_sum = light_condition_variance + road_geometry_variance + road_surface_variance
light_condition_weight = light_condition_variance / variance_sum
road_geometry_weight = road_geometry_variance / variance_sum
road_surface_weight = road_surface_variance / variance_sum

# Print the weights for verification
print("\nWeights for each condition (based on variance):")
print(f"Light Condition Weight: {light_condition_weight:.4f}")
print(f"Road Geometry Weight: {road_geometry_weight:.4f}")
print(f"Road Surface Weight: {road_surface_weight:.4f}")
print(f"Sum of Weights: {light_condition_weight + road_geometry_weight + road_surface_weight:.4f}")

# Read the datasets
accident = pd.read_csv('Data Preprocess/updated_accident.csv')
vehicle = pd.read_csv('Data Preprocess/updated_vehicle.csv')

# Merge the datasets
merged_df = pd.merge(accident, vehicle, on='ACCIDENT_NO', how='inner')

# Create the CATEGORIZED_SEVERITY column
merged_df['CATEGORIZED_SEVERITY'] = merged_df['SEVERITY'].apply(lambda x: 'SIGNIFICANT' if x == 1 or x == 2 else 'NORMAL')

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
merged_df['WEIGHTED_RISK_LIGHT_CONDITION'] = merged_df['SIGNIFICANT_SEVERITY_RISK_LIGHT_CONDITION'] * (light_condition_weight ** 0.5)
merged_df['WEIGHTED_RISK_ROAD_GEOMETRY'] = merged_df['SIGNIFICANT_SEVERITY_RISK_ROAD_GEOMETRY'] * (road_geometry_weight ** 0.5)
merged_df['WEIGHTED_RISK_ROAD_SURFACE'] = merged_df['SIGNIFICANT_SEVERITY_RISK_ROAD_SURFACE'] * (road_surface_weight ** 0.5)

# Drop rows with NaN values in any of the risk columns
merged_df = merged_df.dropna(subset=['WEIGHTED_RISK_LIGHT_CONDITION', 
                                    'WEIGHTED_RISK_ROAD_GEOMETRY', 
                                    'WEIGHTED_RISK_ROAD_SURFACE'])

# Print how many rows were kept after dropping NaN values
print(f"Number of rows after dropping NaN values: {len(merged_df)}")

# Import necessary libraries for the k-NN model
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data for k-NN model using the weighted columns
X = merged_df[['WEIGHTED_RISK_LIGHT_CONDITION', 
              'WEIGHTED_RISK_ROAD_GEOMETRY', 
              'WEIGHTED_RISK_ROAD_SURFACE']]
y = merged_df['CATEGORIZED_SEVERITY']

# Create directory for version2 if it doesn't exist
import os
if not os.path.exists('Task 9 k-NN classification/second kNN version image result'):
    os.makedirs('Task 9 k-NN classification/second kNN version image result')
if not os.path.exists('Task 9 k-NN classification/second kNN version .txt result'):
    os.makedirs('Task 9 k-NN classification/second kNN version .txt result')

# Set the fixed k value for KNN
k = 200
knn = KNeighborsClassifier(n_neighbors=k)

# Perform 5-fold cross-validation
print(f"\nPerforming 5-fold cross-validation with k={k} and weighted features...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
fold_confusion_matrices = []
fold_reports = []

# To store fold results
fold_results = {}

# Iterate through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train the model
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Store results
    fold_accuracies.append(accuracy)
    fold_confusion_matrices.append(cm)
    fold_reports.append(report)
    
    # Store individual fold results
    fold_results[f"Fold {fold+1}"] = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report
    }
    
    # Print fold results
    print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix for each fold
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NORMAL', 'SIGNIFICANT'], 
                yticklabels=['NORMAL', 'SIGNIFICANT'])
    plt.title(f'Fold {fold+1} Confusion Matrix (Accuracy: {accuracy:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'Task 9 k-NN classification/second kNN version image result/knn_confusion_matrix_fold_{fold+1}.png')
    plt.close()

# Calculate average results
avg_accuracy = np.mean(fold_accuracies)
avg_cm = np.mean(fold_confusion_matrices, axis=0).astype(int)

# Create a combined classification report
avg_report = {}
for metric in fold_reports[0].keys():
    if metric not in ['accuracy', 'macro avg', 'weighted avg']:
        avg_report[metric] = {}
        for score in fold_reports[0][metric].keys():
            avg_report[metric][score] = np.mean([fold_reports[i][metric][score] for i in range(5)])

# Print average results
print("\nCross-Validation Results:")
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Std Dev of Accuracy: {np.std(fold_accuracies):.4f}")

# Plot average confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NORMAL', 'SIGNIFICANT'], 
            yticklabels=['NORMAL', 'SIGNIFICANT'])
plt.title(f'Average Confusion Matrix (5-fold CV, Accuracy: {avg_accuracy:.4f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('Task 9 k-NN classification/second kNN version image result/knn_avg_confusion_matrix.png')
plt.close()

# Plot accuracies for each fold
plt.figure(figsize=(10, 6))
plt.bar(range(1, 6), fold_accuracies)
plt.axhline(y=avg_accuracy, color='r', linestyle='-', label=f'Average: {avg_accuracy:.4f}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Weighted KNN Accuracy Across 5 Folds')
plt.xticks(range(1, 6))
plt.ylim(0, 1)
plt.legend()
plt.savefig('Task 9 k-NN classification/second kNN version image result/knn_fold_accuracies.png')
plt.close()

# Save results to a file
with open('Task 9 k-NN classification/second kNN version .txt result/knn_cv_results.txt', 'w') as f:
    f.write(f"5-Fold Cross-Validation Results (k={k}) - Version 2 (Weighted)\n")
    f.write(f"=======================================================\n\n")
    f.write(f"Feature Weights (based on variance):\n")
    f.write(f"Light Condition: {light_condition_weight:.4f}\n")
    f.write(f"Road Geometry: {road_geometry_weight:.4f}\n")
    f.write(f"Road Surface: {road_surface_weight:.4f}\n\n")
    
    f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
    f.write(f"Standard Deviation of Accuracy: {np.std(fold_accuracies):.4f}\n\n")
    
    for fold in range(5):
        f.write(f"Fold {fold+1} Accuracy: {fold_accuracies[fold]:.4f}\n")
        f.write("Classification Report:\n")
        for label in fold_reports[fold]:
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                scores = fold_reports[fold][label]
                f.write(f"  {label}:\n")
                for metric, value in scores.items():
                    f.write(f"    {metric}: {value:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(fold_confusion_matrices[fold]))
        f.write("\n\n")
    
    f.write("Average Confusion Matrix:\n")
    f.write(str(avg_cm))

print("\nCross-validation completed!")
print(f"Results saved to 'Task 9 k-NN classification/second kNN version .txt result/knn_cv_results.txt'")
print(f"Confusion matrices saved to 'Task 9 k-NN classification/second kNN version image result/'")







