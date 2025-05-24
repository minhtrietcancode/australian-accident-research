import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the confusion matrices from the results
standard_cm = np.array([
    [40670, 1048],
    [22329, 1129]
])

optimized_cm = np.array([
    [40692, 1026],
    [22349, 1109]
])

# Set up the figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Define labels
labels = ['NORMAL', 'SIGNIFICANT']

# Plot first confusion matrix - Standard Model
sns.heatmap(standard_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, ax=ax1, cbar=False)
ax1.set_title('Standard Category DT Model\n(5-fold CV, Accuracy: 0.6413)', fontsize=14)
ax1.set_xlabel('Predicted Label', fontsize=12)
ax1.set_ylabel('True Label', fontsize=12)

# Plot second confusion matrix - Optimized Model
sns.heatmap(optimized_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, ax=ax2, cbar=True)
ax2.set_title('Optimized Category DT Model\n(5-fold CV, Accuracy: 0.6414)', fontsize=14)
ax2.set_xlabel('Predicted Label', fontsize=12)
ax2.set_ylabel('True Label', fontsize=12)

# Add an overall title
plt.suptitle('Comparison of Decision Tree Models on Accident Category Classification', fontsize=16)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.88)

# Show the plot
plt.savefig("DT_model_combined_mat.png")