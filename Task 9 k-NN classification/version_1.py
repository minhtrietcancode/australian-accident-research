'''
    - For this task I will implement the k-NN models with the very basic approach first to test which would work better
    - As the k-NN expect the categories to be numerical values, so here is the approach:
      Recall that here is the columns we will choose to use for each condition:
            + CATEGORIZED_LIGHT_CONDITION: for light condition, as the categorized version does not have any biased
              or problem with the new categorized values, it just simply group the things which are actually similar
              so we can use this column for light condition
            + ROAD_GEOMETRY_DESC: with road geometry we have to use the original column as the new CATEGORIZED_ROAD_GEOMETRY
              is just to create broader range of categories for an overview analysis, cannot use it here --> avoid
              any bias
            + CATEGORIZED_ROAD_SURFACE: same reason as the light condition, the new CATEGORIZED_ROAD_SURFACE just rename
              the variable, otherwise it did not do anything that enforces bias

    - Here is how I will assign numerical value for these columns. I will look back at Task 1,2,3 to get the percentage
      of serious / fatal accidents for each category of these 3 conditions, and then assign these probabilities for each
      category in some new columns like SIGNIFICANT_SEVERITY_RISK_LIGHT_CONDITION,
      SIGNIFICANT_SEVERITY_RISK_ROAD_GEOMETRY, SIGNIFICANT_SEVERITY_RISK_ROAD_SURFACE

    - And then with the SEVERITY columns, I will categorize this into just 2 values in a new columns like
      CATEGORIZED_SEVERITY:
        + SIGNIFICANT: SEVERITY = 1 or SEVERITY = 2
        + NORMAL: SEVERITY = 3 or SEVERITY = 4

    - So totally here is what we are going to do with the first version of this k-NN classification model
        + merge two datasets: merged_df like below
        + create 4 more columns: SIGNIFICANT_SEVERITY_RISK_LIGHT_CONDITION, SIGNIFICANT_SEVERITY_RISK_ROAD_GEOMETRY,
                                 SIGNIFICANT_SEVERITY_RISK_ROAD_SURFACE, CATEGORIZED_SEVERITY
        + Use the 80/20 split for train / test
        + And then use the k-NN models to classify the CATEGORIZED_SEVERITY based on the 3 new ***_RISK_*** columns
        + Show the confusion matrix for the result of classification, and also calculate the accuracy score for this
          model and then include that score also somewhere in the confusion matrix to make it be beautiful

    - Note that with the serious / fatal probability, get that from these files
        + task1_4_light_condition_serious_prob.csv
        + task2_4_road_geometry_serious_prob.csv
        + task3_4_road_surface_serious_prob.csv
        + Remember when assign these probabilities to the new columns: scale these prob to be a scale over 100:
          for example: if prob = 0.15 then risk here would be 15 --> do this in order to make the value be more
          beautiful for the model to run

    - Note that here I choose k = 200, and conduct 5-fold cross-validation
      + The reason for choosing k = 200 is because the accuracy score is the highest when k = 200 when i 
        run the code with different k values 
      + 5 folds cross-validation is because it is a good balance as I want the scale of train / test set to be 8/2 

    - Now to summary, the first version of the k-NN model is treating the 3 new ***_RISK_*** columns as the same. However
      in fact here is the true order of these condtitions: Light condition --> Road geometry --> Road surface
      so here is the problem, the k-NN model is not able to capture the order of these conditions, and it is treating
      them as the same --> so for better performance, we need to improve the model by adding the order of these conditions
      into the model and create a weight for each condition --> this will be the next version of the k-NN model
'''
import pandas as pd

# read the dataset
accident = pd.read_csv('Data Preprocess/updated_accident.csv')
vehicle = pd.read_csv('Data Preprocess/updated_vehicle.csv')

# merge the dataset
merged_df = pd.merge(accident, vehicle, on='ACCIDENT_NO', how='inner')

# get the probability of serious / fatal accidents for each category of these 3 conditions
light_condition_prob = pd.read_csv('Task 1 LIGHT_CONDITION VS SEVERITY/task1_4_light_condition_serious_prob.csv')
road_geometry_prob = pd.read_csv('Task 2 ROAD_GEOMETRY vs SEVERITY/task2_4_road_geometry_serious_prob.csv')
road_surface_prob = pd.read_csv('Task 3 ROAD_SURFACE vs SEVERITY/task3_4_road_surface_serious_prob.csv')

'''
# count the columns with NaN values in merged_df
nan_count = merged_df.isna().sum()
print(nan_count)

no NaN values in merged_df
ACCIDENT_NO                    0
LIGHT_CONDITION                0
CATEGORIZED_LIGHT_CONDITION    0
ROAD_GEOMETRY                  0
ROAD_GEOMETRY_DESC             0
CATEGORIZED_ROAD_GEOMETRY      0
SEVERITY                       0
ROAD_SURFACE_TYPE              0
ROAD_SURFACE_TYPE_DESC         0
CATEGORIZED_ROAD_SURFACE       0
dtype: int64
'''

# Now we will create the new CATEGORIZED_SEVERITY column based on the SEVERITY column
merged_df['CATEGORIZED_SEVERITY'] = merged_df['SEVERITY'].apply(lambda x: 'SIGNIFICANT' if x == 1 or x == 2 else 'NORMAL')

# A mapping dictionary for the light condition
light_condition_mapping = {
    'Dark without Lighting': 51.19,
    'Dark with Lighting': 41.62,
    'Daylight': 37.01,
    'Limited Light': 35.19,
    'Unknown': 19.36
}

# A mapping dictionary for the road geometry
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

# A mapping dictionary for the road surface
road_surface_mapping = {
    'Unpaved': 39.15,
    'Gravel': 38.51,
    'Paved': 36.04,
    'Unknown': 17.89
}

# now create the new columns with the mapping dictionaries
merged_df['SIGNIFICANT_SEVERITY_RISK_LIGHT_CONDITION'] = merged_df['CATEGORIZED_LIGHT_CONDITION'].map(light_condition_mapping)
merged_df['SIGNIFICANT_SEVERITY_RISK_ROAD_GEOMETRY'] = merged_df['ROAD_GEOMETRY_DESC'].map(road_geometry_mapping)
merged_df['SIGNIFICANT_SEVERITY_RISK_ROAD_SURFACE'] = merged_df['CATEGORIZED_ROAD_SURFACE'].map(road_surface_mapping)

# Drop rows with NaN values in any of the risk columns
merged_df = merged_df.dropna(subset=['SIGNIFICANT_SEVERITY_RISK_LIGHT_CONDITION', 
                                    'SIGNIFICANT_SEVERITY_RISK_ROAD_GEOMETRY', 
                                    'SIGNIFICANT_SEVERITY_RISK_ROAD_SURFACE'])

# Print how many rows were kept after dropping NaN values
print(f"Number of rows after dropping NaN values: {len(merged_df)}")

# Import necessary libraries for the k-NN model
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data for k-NN model
X = merged_df[['SIGNIFICANT_SEVERITY_RISK_LIGHT_CONDITION', 
              'SIGNIFICANT_SEVERITY_RISK_ROAD_GEOMETRY', 
              'SIGNIFICANT_SEVERITY_RISK_ROAD_SURFACE']]
y = merged_df['CATEGORIZED_SEVERITY']

# Create directory if it doesn't exist
import os
if not os.path.exists('Task 9 k-NN classification/images/version1'):
    os.makedirs('Task 9 k-NN classification/images/version1')
if not os.path.exists('Task 9 k-NN classification/results/version1'):
    os.makedirs('Task 9 k-NN classification/results/version1')

# Set the fixed k value for KNN
k = 200
knn = KNeighborsClassifier(n_neighbors=k)

# Perform 5-fold cross-validation
print(f"\nPerforming 5-fold cross-validation with k={k}...")
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
    plt.savefig(f'Task 9 k-NN classification/images/version1/knn_confusion_matrix_fold_{fold+1}.png')
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
plt.savefig('Task 9 k-NN classification/images/version1/knn_avg_confusion_matrix.png')
plt.close()

# Plot accuracies for each fold
plt.figure(figsize=(10, 6))
plt.bar(range(1, 6), fold_accuracies)
plt.axhline(y=avg_accuracy, color='r', linestyle='-', label=f'Average: {avg_accuracy:.4f}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy Across 5 Folds')
plt.xticks(range(1, 6))
plt.ylim(0, 1)
plt.legend()
plt.savefig('Task 9 k-NN classification/images/version1/knn_fold_accuracies.png')
plt.close()

# Save results to a file
with open('Task 9 k-NN classification/results/version1/knn_cv_results.txt', 'w') as f:
    f.write(f"5-Fold Cross-Validation Results (k={k}) - Version 1\n")
    f.write(f"==============================================\n\n")
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
print(f"Results saved to 'Task 9 k-NN classification/results/version1/knn_cv_results.txt'")
print(f"Confusion matrices saved to 'Task 9 k-NN classification/images/version1/'")










