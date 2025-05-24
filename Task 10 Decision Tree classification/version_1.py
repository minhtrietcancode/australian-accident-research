'''
- With this part, I will use the decision tree classification to classify SEVERITY 
  of the accident based on the 3 conditions:
        + CATEGORIZED_LIGHT_CONDITION
        + ROAD_GEOMETRY_DESC
        + CATEGORIZED_ROAD_SURFACE

- With SEVERITY, I will categorized into broader range with just 2 values:
        + SIGNIFICANT: SEVERITY = 1, 2
        + NORMAL: SEVERITY = 3, 4
        + create new column CATEGORIZED_SEVERITY

- And here would be the first version of the decision tree classification model:
    + Just use the three above columns to classify the CATEGORIZED_SEVERITY 
      --> we have not categorize these 3 columns yet, that would be the next version
      --> remember we could do this as decision tree classification accept categorical data
    + Use 5 folds cross-validation to evaluate the performance of the model
    + Use the accuracy score to evaluate the performance of the model
    + Use the confusion matrix to evaluate the performance of the model 
'''
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import os

# Create directories for results if they don't exist
if not os.path.exists('Task 10 Decision Tree classification/images/version1'):
    os.makedirs('Task 10 Decision Tree classification/images/version1')
if not os.path.exists('Task 10 Decision Tree classification/results/version1'):
    os.makedirs('Task 10 Decision Tree classification/results/version1')

# Read the dataset
accident = pd.read_csv('Data Preprocess/updated_accident.csv')
vehicle = pd.read_csv('Data Preprocess/updated_vehicle.csv')

# Merge the dataset
merged_df = pd.merge(accident, vehicle, on='ACCIDENT_NO', how='inner')

# Create the new CATEGORIZED_SEVERITY column based on the SEVERITY column
merged_df['CATEGORIZED_SEVERITY'] = merged_df['SEVERITY'].apply(lambda x: 'SIGNIFICANT' if x == 1 or x == 2 else 'NORMAL')

# Print some basic information about the dataset
print(f"Dataset shape: {merged_df.shape}")
print("\nFeature distribution:")
for feature in ['CATEGORIZED_LIGHT_CONDITION', 'ROAD_GEOMETRY_DESC', 'CATEGORIZED_ROAD_SURFACE']:
    print(f"\n{feature} value counts:")
    print(merged_df[feature].value_counts())

print("\nTarget distribution:")
print(merged_df['CATEGORIZED_SEVERITY'].value_counts())

# Check for missing values in the features we're using
features = ['CATEGORIZED_LIGHT_CONDITION', 'ROAD_GEOMETRY_DESC', 'CATEGORIZED_ROAD_SURFACE']
print("\nMissing values in features:")
print(merged_df[features].isnull().sum())

# Drop rows with missing values in our features or target
df_model = merged_df.dropna(subset=features + ['CATEGORIZED_SEVERITY'])
print(f"\nDataset shape after dropping missing values: {df_model.shape}")

# Prepare the data for the model
X = df_model[features]
y = df_model['CATEGORIZED_SEVERITY']

# Using one-hot encoding for categorical features
X_encoded = pd.get_dummies(X, drop_first=False)

# Create the decision tree classifier (with no depth limitation for the main model)
dt_classifier = DecisionTreeClassifier(random_state=42)

# Implement proper 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store results
fold_accuracies = []
fold_confusion_matrices = []
fold_reports = []
y_true_all = []
y_pred_all = []

# To store fold results
fold_results = {}

# Perform cross-validation manually to properly track test results
print("\nPerforming 5-fold cross-validation...")
for fold, (train_idx, test_idx) in enumerate(cv.split(X_encoded, y)):
    # Split the data for this fold
    X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train the model on the training data
    dt_classifier.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = dt_classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Store results
    fold_accuracies.append(accuracy)
    fold_confusion_matrices.append(cm)
    fold_reports.append(report)
    
    # Store the true and predicted values for confusion matrix later
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)
    
    # Store individual fold results
    fold_results[f"Fold {fold+1}"] = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report
    }
    
    # Print fold results
    print(f"Fold {fold+1} Accuracy: {accuracy:.4f} (Test size: {len(y_test)})")
    print(f"Fold {fold+1} Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix for each fold
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NORMAL', 'SIGNIFICANT'], 
                yticklabels=['NORMAL', 'SIGNIFICANT'])
    plt.title(f'Fold {fold+1} Confusion Matrix (Accuracy: {accuracy:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'Task 10 Decision Tree classification/images/version1/dt_confusion_matrix_fold_{fold+1}.png')
    plt.close()

# Convert to numpy array for convenience
y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# Calculate overall metrics
overall_accuracy = accuracy_score(y_true_all, y_pred_all)
overall_conf_matrix = confusion_matrix(y_true_all, y_pred_all)
class_report = classification_report(y_true_all, y_pred_all)

# Calculate average confusion matrix
avg_cm = np.mean(fold_confusion_matrices, axis=0).astype(int)

# Print the results
print("\n===== Decision Tree Classification Results =====")
print(f"Cross-validation accuracy scores: {fold_accuracies}")
print(f"Mean accuracy: {np.mean(fold_accuracies):.4f} (±{np.std(fold_accuracies):.4f})")
print(f"Overall accuracy on test folds: {overall_accuracy:.4f}")

print("\nAverage Confusion Matrix:")
print(avg_cm)

print("\nClassification Report:")
print(class_report)

# Train a model on the full dataset to get feature importance
dt_classifier.fit(X_encoded, y)

# Feature importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': dt_classifier.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(feature_importance)

# Plot average confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NORMAL', 'SIGNIFICANT'], 
            yticklabels=['NORMAL', 'SIGNIFICANT'])
plt.title(f'Average Confusion Matrix (5-fold CV, Accuracy: {np.mean(fold_accuracies):.4f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('Task 10 Decision Tree classification/images/version1/dt_avg_confusion_matrix.png')
plt.close()

# Plot accuracies for each fold
plt.figure(figsize=(10, 6))
plt.bar(range(1, 6), fold_accuracies)
plt.axhline(y=np.mean(fold_accuracies), color='r', linestyle='-', label=f'Average: {np.mean(fold_accuracies):.4f}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy Across 5 Folds')
plt.xticks(range(1, 6))
plt.ylim(0, 1)
plt.legend()
plt.savefig('Task 10 Decision Tree classification/images/version1/dt_fold_accuracies.png')
plt.close()

# Visualize the decision tree (limited to depth 3 for better visualization)
# Create a separate tree with max_depth=3 for visualization purposes
limited_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
limited_dt.fit(X_encoded, y)

# Check the actual max depth of the limited tree
print(f"\nActual depth of limited tree: {limited_dt.get_depth()}")

plt.figure(figsize=(20, 10))
tree.plot_tree(limited_dt, feature_names=X_encoded.columns, class_names=limited_dt.classes_, 
              filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree (Limited to Depth 3)')
plt.tight_layout()
plt.savefig('Task 10 Decision Tree classification/images/version1/decision_tree.png')
plt.close()

# Save the results to a file
with open('Task 10 Decision Tree classification/results/version1/dt_cv_results.txt', 'w') as f:
    f.write("===== Decision Tree Classification Results =====\n")
    f.write(f"Dataset size: {len(df_model)}\n")
    f.write(f"Test set size (total across all folds): {len(y_true_all)}\n\n")
    
    f.write("Cross-validation Accuracy Scores:\n")
    for i, acc in enumerate(fold_accuracies):
        f.write(f"Fold {i+1}: {acc:.4f}\n")
    
    f.write(f"\nMean accuracy: {np.mean(fold_accuracies):.4f} (±{np.std(fold_accuracies):.4f})\n")
    f.write(f"Overall accuracy on test folds: {overall_accuracy:.4f}\n\n")
    
    f.write("Individual Fold Results:\n")
    for fold in fold_results:
        f.write(f"\n{fold}:\n")
        f.write(f"  Accuracy: {fold_results[fold]['accuracy']:.4f}\n")
        f.write(f"  Confusion Matrix:\n")
        f.write(f"  {fold_results[fold]['confusion_matrix']}\n")
    
    f.write("\nAverage Confusion Matrix:\n")
    f.write(f"{avg_cm}\n\n")
    
    f.write("Overall Confusion Matrix (Test Folds Only):\n")
    f.write(f"{overall_conf_matrix}\n\n")
    
    f.write("Classification Report:\n")
    f.write(f"{class_report}\n\n")
    
    f.write("Feature Importance:\n")
    for idx, row in feature_importance.iterrows():
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")

print("\nAll results saved to files in Task 10 Decision Tree classification/version1")

