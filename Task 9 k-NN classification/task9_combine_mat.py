import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Average confusion matrices from your results
unweighted_cm = np.array([
    [38211, 3507],
    [20591, 2867]
])

weighted_cm = np.array([
    [38187, 3531],
    [20572, 2886]
])

# Accuracy scores
unweighted_acc = 0.6303
weighted_acc = 0.6302

# Create figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Function to plot confusion matrix
def plot_confusion_matrix(cm, ax, title, acc):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'{title}\n(5-fold CV, Accuracy: {acc:.4f})')
    ax.set_xticklabels(['NORMAL', 'SIGNIFICANT'])
    ax.set_yticklabels(['NORMAL', 'SIGNIFICANT'])

# Plot the first confusion matrix (Unweighted)
plot_confusion_matrix(unweighted_cm, ax1, 'Unweighted kNN Model', unweighted_acc)

# Plot the second confusion matrix (Weighted)
plot_confusion_matrix(weighted_cm, ax2, 'Weighted kNN Model', weighted_acc)

# Adjust layout
plt.tight_layout()
plt.savefig('combined_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()
