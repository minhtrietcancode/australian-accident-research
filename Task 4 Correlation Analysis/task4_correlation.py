import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
import matplotlib.patheffects as path_effects


# Function to calculate entropy H(X)
def entropy(x):
    """
    Calculate Shannon entropy H(X) for a discrete variable X
    H(X) = -sum(P(x_i) * log2(P(x_i)))
    """
    # Count occurrences of each value
    counter = Counter(x)
    total = len(x)

    # Calculate entropy
    h = 0
    for count in counter.values():
        prob = count / total
        h -= prob * np.log2(prob)

    return h

# Function to calculate conditional entropy H(Y|X)
def conditional_entropy(x, y):
    """
    Calculate conditional entropy H(Y|X)
    H(Y|X) = sum_i P(X=x_i) * H(Y|X=x_i)
    """
    # Get unique values of X
    x_values = np.unique(x)

    # Calculate conditional entropy
    h_y_given_x = 0

    for x_val in x_values:
        # Find indices where X = x_val
        indices = np.where(x == x_val)[0]

        # Calculate probability P(X=x_val)
        p_x = len(indices) / len(x)

        # Get corresponding Y values
        y_given_x = y[indices]

        # Calculate H(Y|X=x_val)
        h_y_given_x_val = entropy(y_given_x)

        # Add weighted entropy to total
        h_y_given_x += p_x * h_y_given_x_val

    return h_y_given_x

# Function to calculate mutual information MI(X, Y)
def mutual_information(x, y):
    """
    Calculate mutual information MI(X, Y)
    MI(X, Y) = H(Y) - H(Y|X) = H(X) - H(X|Y)
    """
    h_y = entropy(y)
    h_y_given_x = conditional_entropy(x, y)

    return h_y - h_y_given_x

# Function to calculate normalized mutual information NMI(X, Y)
def normalized_mutual_information(x, y):
    """
    Calculate normalized mutual information NMI(X, Y)
    NMI(X, Y) = MI(X, Y) / min(H(X), H(Y))
    """
    mi = mutual_information(x, y)
    h_x = entropy(x)
    h_y = entropy(y)

    # Avoid division by zero
    denominator = min(h_x, h_y)
    if denominator == 0:
        return 0

    return mi / denominator

# Function to calculate multivariate mutual information for 2 variables with SEVERITY
def multivariate_mi_2var(x, y, z):
    """
    Calculate multivariate mutual information I(X,Y;Z)
    I(X,Y;Z) = I(X;Z) + I(Y;Z|X)
    """
    # Calculate I(X;Z) - mutual information between X and Z
    mi_x_z = mutual_information(x, z)

    # To calculate I(Y;Z|X), we need:
    # I(Y;Z|X) = H(Z|X) - H(Z|X,Y)

    # Calculate H(Z|X)
    h_z_given_x = conditional_entropy(x, z)

    # Calculate H(Z|X,Y)
    # We'll create a joint variable XY to represent the combination of X and Y
    xy = np.array([f"{x[i]}_{y[i]}" for i in range(len(x))])
    h_z_given_xy = conditional_entropy(xy, z)

    # I(Y;Z|X) = H(Z|X) - H(Z|X,Y)
    mi_y_z_given_x = h_z_given_x - h_z_given_xy

    # Final multivariate MI
    mmi = mi_x_z + mi_y_z_given_x

    # Normalize by H(Z)
    h_z = entropy(z)
    if h_z == 0:
        return 0

    return mmi / h_z

# Function to calculate multivariate mutual information for 3 variables with SEVERITY
def multivariate_mi_3var(x, y, w, z):
    """
    Calculate multivariate mutual information I(X,Y,W;Z)
    I(X,Y,W;Z) = I(X;Z) + I(Y;Z|X) + I(W;Z|X,Y)
    """
    # Calculate I(X;Z)
    mi_x_z = mutual_information(x, z)

    # Calculate I(Y;Z|X) = H(Z|X) - H(Z|X,Y)
    h_z_given_x = conditional_entropy(x, z)
    xy = np.array([f"{x[i]}_{y[i]}" for i in range(len(x))])
    h_z_given_xy = conditional_entropy(xy, z)
    mi_y_z_given_x = h_z_given_x - h_z_given_xy

    # Calculate I(W;Z|X,Y) = H(Z|X,Y) - H(Z|X,Y,W)
    h_z_given_xy = h_z_given_xy  # Already calculated above
    xyw = np.array([f"{xy[i]}_{w[i]}" for i in range(len(x))])
    h_z_given_xyw = conditional_entropy(xyw, z)
    mi_w_z_given_xy = h_z_given_xy - h_z_given_xyw

    # Final multivariate MI
    mmi = mi_x_z + mi_y_z_given_x + mi_w_z_given_xy

    # Normalize by H(Z)
    h_z = entropy(z)
    if h_z == 0:
        return 0

    return mmi / h_z

# Load the data
print("Loading datasets...")
try:
    accident_df = pd.read_csv('accident.csv')
    vehicle_df = pd.read_csv('vehicle.csv')

    # Merge datasets on ACCIDENT_NO
    print("Merging datasets...")
    merged_df = pd.merge(accident_df, vehicle_df, on='ACCIDENT_NO', how='inner')

    # Select columns of interest
    data = merged_df[['LIGHT_CONDITION', 'ROAD_GEOMETRY', 'ROAD_SURFACE_TYPE', 'SEVERITY']].copy()

    # Handle missing values
    data = data.dropna()
    print(f"Dataset after cleaning: {len(data)} rows")

    # For calculation clarity, convert categorical columns to category type (which assigns numeric codes)
    features = ['LIGHT_CONDITION', 'ROAD_GEOMETRY', 'ROAD_SURFACE_TYPE', 'SEVERITY']
    for col in features:
        data[col] = data[col].astype('category').cat.codes

    # Calculate normalized mutual information for individual factors with SEVERITY
    print("\nCalculating individual normalized mutual information...")
    light_severity_nmi = normalized_mutual_information(data['LIGHT_CONDITION'].values, data['SEVERITY'].values)
    geometry_severity_nmi = normalized_mutual_information(data['ROAD_GEOMETRY'].values, data['SEVERITY'].values)
    surface_severity_nmi = normalized_mutual_information(data['ROAD_SURFACE_TYPE'].values, data['SEVERITY'].values)

    print(f"NMI(LIGHT_CONDITION, SEVERITY) = {light_severity_nmi:.4f}")
    print(f"NMI(ROAD_GEOMETRY, SEVERITY) = {geometry_severity_nmi:.4f}")
    print(f"NMI(ROAD_SURFACE_TYPE, SEVERITY) = {surface_severity_nmi:.4f}")

    # Calculate multivariate mutual information for pairwise combinations
    print("\nCalculating pairwise multivariate mutual information...")
    light_geometry_severity_mmi = multivariate_mi_2var(
        data['LIGHT_CONDITION'].values, data['ROAD_GEOMETRY'].values, data['SEVERITY'].values)
    light_surface_severity_mmi = multivariate_mi_2var(
        data['LIGHT_CONDITION'].values, data['ROAD_SURFACE_TYPE'].values, data['SEVERITY'].values)
    geometry_surface_severity_mmi = multivariate_mi_2var(
        data['ROAD_GEOMETRY'].values, data['ROAD_SURFACE_TYPE'].values, data['SEVERITY'].values)

    print(f"MMI(LIGHT_CONDITION, ROAD_GEOMETRY; SEVERITY) = {light_geometry_severity_mmi:.4f}")
    print(f"MMI(LIGHT_CONDITION, ROAD_SURFACE_TYPE; SEVERITY) = {light_surface_severity_mmi:.4f}")
    print(f"MMI(ROAD_GEOMETRY, ROAD_SURFACE_TYPE; SEVERITY) = {geometry_surface_severity_mmi:.4f}")

    # Calculate multivariate mutual information for all three factors
    print("\nCalculating multivariate mutual information for all three factors...")
    all_factors_severity_mmi = multivariate_mi_3var(
        data['LIGHT_CONDITION'].values,
        data['ROAD_GEOMETRY'].values,
        data['ROAD_SURFACE_TYPE'].values,
        data['SEVERITY'].values
    )

    print(f"MMI(LIGHT_CONDITION, ROAD_GEOMETRY, ROAD_SURFACE_TYPE; SEVERITY) = {all_factors_severity_mmi:.4f}")

    # Compare information gain
    max_individual_nmi = max(light_severity_nmi, geometry_severity_nmi, surface_severity_nmi)
    max_pair_mmi = max(light_geometry_severity_mmi, light_surface_severity_mmi, geometry_surface_severity_mmi)

    print("\nComparison of Information Gain:")
    print(f"- Maximum Individual NMI: {max_individual_nmi:.4f}")
    print(f"- Maximum Pairwise MMI: {max_pair_mmi:.4f}")
    print(f"- Three-Factor MMI: {all_factors_severity_mmi:.4f}")

    if all_factors_severity_mmi > max_pair_mmi:
        info_gain = all_factors_severity_mmi - max_pair_mmi
        percentage_gain = (info_gain / max_pair_mmi) * 100
        print(
            f"\nThe combination of all three factors provides more information about SEVERITY than any pair of factors.")
        print(f"Information gain: {info_gain:.4f} (or {percentage_gain:.2f}% increase)")
    else:
        print(
            "\nThe combination of all three factors doesn't provide substantially more information than the best pair.")

    if all_factors_severity_mmi > max_individual_nmi:
        info_gain = all_factors_severity_mmi - max_individual_nmi
        percentage_gain = (info_gain / max_individual_nmi) * 100
        print(f"Information gain over best individual feature: {info_gain:.4f} (or {percentage_gain:.2f}% increase)")

    # Create visualization
    print("\nCreating visualization...")
    # Prepare data for plotting
    plot_data = [
        {'Factors': 'Light Condition', 'Type': 'Single', 'MI': light_severity_nmi},
        {'Factors': 'Road Geometry', 'Type': 'Single', 'MI': geometry_severity_nmi},
        {'Factors': 'Road Surface', 'Type': 'Single', 'MI': surface_severity_nmi},
        {'Factors': 'Light + Geometry', 'Type': 'Pairwise', 'MI': light_geometry_severity_mmi},
        {'Factors': 'Light + Surface', 'Type': 'Pairwise', 'MI': light_surface_severity_mmi},
        {'Factors': 'Geometry + Surface', 'Type': 'Pairwise', 'MI': geometry_surface_severity_mmi},
        {'Factors': 'Light + Geometry + Surface', 'Type': 'Full', 'MI': all_factors_severity_mmi}
    ]

    plot_df = pd.DataFrame(plot_data)

    # Sort by MI value for better visualization
    plot_df = plot_df.sort_values('MI')

    # Create color mapping
    color_map = {'Single': 'blue', 'Pairwise': 'orange', 'Full': 'red'}

    # Create the figure
    plt.figure(figsize=(12, 8))

    # Plot horizontal bar chart
    bars = plt.barh(plot_df['Factors'], plot_df['MI'], color=[color_map[t] for t in plot_df['Type']])

    # Add labels and title
    plt.xlabel('Normalized Mutual Information', fontsize=14)
    plt.title('Mutual Information between Road Factors and Accident Severity', fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='blue', label='Single Factor'),
        Patch(facecolor='orange', label='Pairwise Factors'),
        Patch(facecolor='red', label='All Three Factors')
    ]
    plt.legend(handles=legend_elements)

    # Add values on bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.4f}',
                 va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('final mutual_information_analysis.png', dpi=300)

    print("Analysis complete! Results saved to 'mutual_information_analysis.png'")

except Exception as e:
    print(f"An error occurred: {e}")


# NOTE THAT THE ABOVE CODE IS GOOD, HOWEVER WITH THE VISUALIZATION IT IS NOT REALLY
# BEAUTIFUL SO THIS PART IS TO USE THE RESULT OF MUTUAL INFORMATION AFTER RUNNING THE ABOVE CODE AND
# THEN CREATE THE VISUALIZATION BASED ON THESE NUMBERS
# Prepare data for plotting
plot_data = [
    {'Factors': 'Light Condition', 'Type': 'Single', 'MI': light_severity_nmi},
    {'Factors': 'Road Geometry', 'Type': 'Single', 'MI': geometry_severity_nmi},
    {'Factors': 'Road Surface', 'Type': 'Single', 'MI': surface_severity_nmi},
    {'Factors': 'Light + Geometry', 'Type': 'Pairwise', 'MI': light_geometry_severity_mmi},
    {'Factors': 'Light + Surface', 'Type': 'Pairwise', 'MI': light_surface_severity_mmi},
    {'Factors': 'Geometry + Surface', 'Type': 'Pairwise', 'MI': geometry_surface_severity_mmi},
    {'Factors': 'All Three Factors', 'Type': 'Full', 'MI': all_factors_severity_mmi}
]

plot_df = pd.DataFrame(plot_data)

# Sort by MI value and factor type for better visualization
# This gives a more logical grouping and progression
plot_df['Type_order'] = pd.Categorical(plot_df['Type'], categories=['Single', 'Pairwise', 'Full'], ordered=True)
plot_df = plot_df.sort_values(['Type_order', 'MI'])

# Create a more modern and visually appealing figure
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 9))

# Create better color palette with higher contrast
colors = {'Single': '#1f77b4', 'Pairwise': '#ff7f0e', 'Full': '#d62728'}

# Plot horizontal bar chart with improved aesthetics
bars = ax.barh(plot_df['Factors'], plot_df['MI'],
               color=[colors[t] for t in plot_df['Type']],
               height=0.6,
               alpha=0.85)

# Improve grid appearance
ax.grid(axis='x', linestyle='--', alpha=0.4, color='gray')

# Add descriptive annotations for each bar
for i, bar in enumerate(bars):
    value = bar.get_width()
    text = ax.text(value + 0.0005, bar.get_y() + bar.get_height()/2,
                  f'{value:.4f}',
                  va='center',
                  ha='left',
                  fontweight='bold',
                  fontsize=10)
    # Add text outline for better visibility
    text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

# Create a better legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['Single'], label='Single Factor', alpha=0.85),
    Patch(facecolor=colors['Pairwise'], label='Pairwise Factors', alpha=0.85),
    Patch(facecolor=colors['Full'], label='All Three Factors', alpha=0.85)
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True,
          framealpha=0.95, edgecolor='gray', fontsize=10)

# Add a visual indicator for the best performer
best_idx = plot_df['MI'].idxmax()
best_factor = plot_df.loc[best_idx, 'Factors']
best_value = plot_df.loc[best_idx, 'MI']

# Highlight the best performer
for i, bar in enumerate(bars):
    if i == best_idx - plot_df.index[0]:  # Account for sorted index
        bar.set_alpha(1.0)
        bar.set_edgecolor('black')
        bar.set_linewidth(1.5)

# Add titles and labels with better formatting
ax.set_xlabel('Normalized Mutual Information', fontsize=14, fontweight='bold')
ax.set_title('Mutual Information between Road Factors and Accident Severity',
             fontsize=18, fontweight='bold', pad=20)

# Add a subtitle with key insights
max_individual = plot_df[plot_df['Type'] == 'Single']['MI'].max()
info_gain = all_factors_severity_mmi - max_individual
percentage_gain = (info_gain / max_individual) * 100

fig.text(0.5, 0.91,
         f"Combining all factors provides {percentage_gain:.1f}% more information than the best individual factor",
         ha='center', fontsize=12, style='italic', color='#555555')

# Add an informative caption
fig.text(0.5, 0.02,
         "Higher values indicate stronger relationship between factors and accident severity.\n"
         "Information gain measures how much better we can predict severity with multiple factors.",
         ha='center', fontsize=10, color='#555555')

# Add a y-axis label to clarify what each row represents
ax.set_ylabel('Road Factors', fontsize=14, fontweight='bold')

# Improve tick labels
ax.tick_params(axis='both', which='major', labelsize=11)

# Set x-axis limits with some padding
ax.set_xlim(0, plot_df['MI'].max() * 1.15)

# Add background shading for different factor types
prev_type = None
start_idx = 0

for i, (idx, row) in enumerate(plot_df.iterrows()):
    if prev_type is None or prev_type != row['Type']:
        if prev_type is not None:
            ax.axhspan(i - 0.5, start_idx - 0.5, color=colors[prev_type], alpha=0.05)
        prev_type = row['Type']
        start_idx = i
    if i == len(plot_df) - 1:  # Last row
        ax.axhspan(i + 0.5, start_idx - 0.5, color=colors[prev_type], alpha=0.05)

# Adjust layout
fig.tight_layout(rect=[0, 0.04, 1, 0.90])  # Make room for the subtitle and caption

# Save the figure with higher resolution
plt.savefig('improved_mutual_information_analysis.png', dpi=300, bbox_inches='tight')

print("Enhanced visualization saved as 'improved_mutual_information_analysis.png'")

### IGNORE THE OTHER VISUALIZATION, JUST CARE ABOUT THE improved_mutual_information_analysis.png