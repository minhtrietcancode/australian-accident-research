import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch


def entropy(x):
    """
    Calculate Shannon entropy H(X) for a discrete variable X
    H(X) = -sum(P(x_i) * log2(P(x_i)))
    """
    counter = Counter(x)
    total = len(x)
    
    h = 0
    for count in counter.values():
        prob = count / total
        h -= prob * np.log2(prob)
    
    return h


def conditional_entropy(x, y):
    """
    Calculate conditional entropy H(Y|X)
    H(Y|X) = sum_i P(X=x_i) * H(Y|X=x_i)
    """
    x_values = np.unique(x)
    h_y_given_x = 0
    
    for x_val in x_values:
        indices = np.where(x == x_val)[0]
        p_x = len(indices) / len(x)
        y_given_x = y[indices]
        h_y_given_x_val = entropy(y_given_x)
        h_y_given_x += p_x * h_y_given_x_val
    
    return h_y_given_x


def mutual_information(x, y):
    """
    Calculate mutual information MI(X, Y)
    MI(X, Y) = H(Y) - H(Y|X) = H(X) - H(X|Y)
    """
    h_y = entropy(y)
    h_y_given_x = conditional_entropy(x, y)
    return h_y - h_y_given_x


def normalized_mutual_information(x, y):
    """
    Calculate normalized mutual information NMI(X, Y)
    NMI(X, Y) = MI(X, Y) / min(H(X), H(Y))
    """
    mi = mutual_information(x, y)
    h_x = entropy(x)
    h_y = entropy(y)
    
    denominator = min(h_x, h_y)
    if denominator == 0:
        return 0
    
    return mi / denominator


def multivariate_mi_2var(x, y, z):
    """
    Calculate multivariate mutual information I(X,Y;Z)
    I(X,Y;Z) = I(X;Z) + I(Y;Z|X)
    """
    # Calculate I(X;Z)
    mi_x_z = mutual_information(x, z)
    
    # Calculate I(Y;Z|X) = H(Z|X) - H(Z|X,Y)
    h_z_given_x = conditional_entropy(x, z)
    xy = np.array([f"{x[i]}_{y[i]}" for i in range(len(x))])
    h_z_given_xy = conditional_entropy(xy, z)
    mi_y_z_given_x = h_z_given_x - h_z_given_xy
    
    # Final multivariate MI
    mmi = mi_x_z + mi_y_z_given_x
    
    # Normalize by H(Z)
    h_z = entropy(z)
    if h_z == 0:
        return 0
    
    return mmi / h_z


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


def create_enhanced_visualization(results_dict):
    """
    Create an enhanced visualization of mutual information results
    """
    # Prepare data for plotting
    plot_data = [
        {'Factors': 'Light Condition', 'Type': 'Single', 'MI': results_dict['light_severity_nmi']},
        {'Factors': 'Road Geometry', 'Type': 'Single', 'MI': results_dict['geometry_severity_nmi']},
        {'Factors': 'Road Surface', 'Type': 'Single', 'MI': results_dict['surface_severity_nmi']},
        {'Factors': 'Light + Geometry', 'Type': 'Pairwise', 'MI': results_dict['light_geometry_severity_mmi']},
        {'Factors': 'Light + Surface', 'Type': 'Pairwise', 'MI': results_dict['light_surface_severity_mmi']},
        {'Factors': 'Geometry + Surface', 'Type': 'Pairwise', 'MI': results_dict['geometry_surface_severity_mmi']},
        {'Factors': 'All Three Factors', 'Type': 'Full', 'MI': results_dict['all_factors_severity_mmi']}
    ]
    
    plot_df = pd.DataFrame(plot_data)
    
    # Sort by type and MI value for better visualization
    plot_df['Type_order'] = pd.Categorical(plot_df['Type'], categories=['Single', 'Pairwise', 'Full'], ordered=True)
    plot_df = plot_df.sort_values(['Type_order', 'MI'])
    
    # Create figure with modern styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Define color palette
    colors = {'Single': '#1f77b4', 'Pairwise': '#ff7f0e', 'Full': '#d62728'}
    
    # Create horizontal bar chart
    bars = ax.barh(plot_df['Factors'], plot_df['MI'],
                   color=[colors[t] for t in plot_df['Type']],
                   height=0.6, alpha=0.85)
    
    # Improve grid appearance
    ax.grid(axis='x', linestyle='--', alpha=0.4, color='gray')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        value = bar.get_width()
        text = ax.text(value + 0.0005, bar.get_y() + bar.get_height()/2,
                      f'{value:.4f}', va='center', ha='left',
                      fontweight='bold', fontsize=10)
        text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    
    # Highlight the best performer
    best_idx = plot_df['MI'].idxmax()
    best_bar_idx = plot_df.index.get_loc(best_idx)
    bars[best_bar_idx].set_alpha(1.0)
    bars[best_bar_idx].set_edgecolor('black')
    bars[best_bar_idx].set_linewidth(1.5)
    
    # Create legend
    legend_elements = [
        Patch(facecolor=colors['Single'], label='Single Factor', alpha=0.85),
        Patch(facecolor=colors['Pairwise'], label='Pairwise Factors', alpha=0.85),
        Patch(facecolor=colors['Full'], label='All Three Factors', alpha=0.85)
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              framealpha=0.95, edgecolor='gray', fontsize=10)
    
    # Add titles and labels
    ax.set_xlabel('Normalized Mutual Information', fontsize=14, fontweight='bold')
    ax.set_ylabel('Road Factors', fontsize=14, fontweight='bold')
    ax.set_title('Mutual Information between Road Factors and Accident Severity',
                 fontsize=18, fontweight='bold', pad=20)
    
    # Calculate and add subtitle with key insights
    max_individual = max(results_dict['light_severity_nmi'], 
                        results_dict['geometry_severity_nmi'], 
                        results_dict['surface_severity_nmi'])
    info_gain = results_dict['all_factors_severity_mmi'] - max_individual
    percentage_gain = (info_gain / max_individual) * 100
    
    fig.text(0.5, 0.91,
             f"Combining all factors provides {percentage_gain:.1f}% more information than the best individual factor",
             ha='center', fontsize=12, style='italic', color='#555555')
    
    # Add informative caption
    fig.text(0.5, 0.02,
             "Higher values indicate stronger relationship between factors and accident severity.\n"
             "Information gain measures how much better we can predict severity with multiple factors.",
             ha='center', fontsize=10, color='#555555')
    
    # Improve appearance
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_xlim(0, plot_df['MI'].max() * 1.15)
    
    # Adjust layout and save
    fig.tight_layout(rect=[0, 0.04, 1, 0.90])
    return fig


def main():
    """
    Main function to run the mutual information analysis
    """
    print("Loading datasets...")
    try:
        # Load and merge datasets
        accident_df = pd.read_csv('Data Preprocess/updated_accident.csv')
        vehicle_df = pd.read_csv('Data Preprocess/updated_vehicle.csv')
        
        print("Merging datasets...")
        merged_df = pd.merge(accident_df, vehicle_df, on='ACCIDENT_NO', how='inner')
        
        # Select and clean data
        data = merged_df[['LIGHT_CONDITION', 'ROAD_GEOMETRY', 'ROAD_SURFACE_TYPE', 'SEVERITY']].copy()
        data = data.dropna()
        print(f"Dataset after cleaning: {len(data)} rows")
        
        # Convert categorical columns to numeric codes
        features = ['LIGHT_CONDITION', 'ROAD_GEOMETRY', 'ROAD_SURFACE_TYPE', 'SEVERITY']
        for col in features:
            data[col] = data[col].astype('category').cat.codes
        
        # Calculate individual normalized mutual information
        print("\nCalculating individual normalized mutual information...")
        light_severity_nmi = normalized_mutual_information(data['LIGHT_CONDITION'].values, data['SEVERITY'].values)
        geometry_severity_nmi = normalized_mutual_information(data['ROAD_GEOMETRY'].values, data['SEVERITY'].values)
        surface_severity_nmi = normalized_mutual_information(data['ROAD_SURFACE_TYPE'].values, data['SEVERITY'].values)
        
        print(f"NMI(LIGHT_CONDITION, SEVERITY) = {light_severity_nmi:.4f}")
        print(f"NMI(ROAD_GEOMETRY, SEVERITY) = {geometry_severity_nmi:.4f}")
        print(f"NMI(ROAD_SURFACE_TYPE, SEVERITY) = {surface_severity_nmi:.4f}")
        
        # Calculate pairwise multivariate mutual information
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
        
        # Calculate three-factor multivariate mutual information
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
            print(f"\nThe combination of all three factors provides more information about SEVERITY than any pair of factors.")
            print(f"Information gain: {info_gain:.4f} (or {percentage_gain:.2f}% increase)")
        else:
            print("\nThe combination of all three factors doesn't provide substantially more information than the best pair.")
        
        if all_factors_severity_mmi > max_individual_nmi:
            info_gain = all_factors_severity_mmi - max_individual_nmi
            percentage_gain = (info_gain / max_individual_nmi) * 100
            print(f"Information gain over best individual feature: {info_gain:.4f} (or {percentage_gain:.2f}% increase)")
        
        # Store results for visualization
        results = {
            'light_severity_nmi': light_severity_nmi,
            'geometry_severity_nmi': geometry_severity_nmi,
            'surface_severity_nmi': surface_severity_nmi,
            'light_geometry_severity_mmi': light_geometry_severity_mmi,
            'light_surface_severity_mmi': light_surface_severity_mmi,
            'geometry_surface_severity_mmi': geometry_surface_severity_mmi,
            'all_factors_severity_mmi': all_factors_severity_mmi
        }
        
        # Create enhanced visualization
        print("\nCreating enhanced visualization...")
        fig = create_enhanced_visualization(results)
        plt.savefig('Task 4 Correlation Analysis/mutual_information_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Analysis complete! Enhanced visualization saved as 'mutual_information_analysis.png'")
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()