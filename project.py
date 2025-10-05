import pandas as pd  # Import pandas library for data manipulation
import matplotlib.pyplot as plt  # Import matplotlib for data visualization
from matplotlib.colors import ListedColormap  # Import ListedColormap for discrete colors

# Set Times New Roman as default font for all plots
plt.rcParams['font.family'] = 'Times New Roman'  # Set font family to Times New Roman




# Step 1: Data Processing
################################################
# Load CSV data into pandas DataFrame
df = pd.read_csv('data.csv')  # Read CSV file and store in DataFrame variable df

# Define discrete colors for each step (13 distinct colors)
step_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',  # Define list of distinct colors
               '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',  # for each maintenance step
               '#008080', '#e6beff', '#9a6324']  # Total of 13 colors for 13 steps
discrete_cmap = ListedColormap(step_colors)  # Create discrete colormap from color list

# Show dataset information
print("Shape of dataset:", df.shape)  # Print number of rows and columns
print("\nDataset preview:")  # Print header for preview section
print(df.head())  # Display first 5 rows of the dataset
print("\nColumn information:")  # Print header for column info
print(df.info())  # Display data types and null value information



# Step 2: Data Visualization
#############################################
print("\n" + "-"*50)  # Print separator line with 60 dashes
print("Step 2: Statistical Analysis and Visualization")  # Print section title
print("-"*50)  # Print separator line

# Display descriptive statistics
print("\nDescriptive Statistics:")  # Print header for statistics
print(df.describe())  # Calculate and display mean, std, min, max, quartiles

# 3D scatter plot showing all maintenance steps
fig1 = plt.figure(figsize=(10, 7))  # Create new figure with 10x7 inch size
ax1 = fig1.add_subplot(111, projection='3d')  # Add 3D axis to the figure
scatter_plot = ax1.scatter(df['X'], df['Y'], df['Z'], c=df['Step']-1,  # Create 3D scatter plot with X, Y, Z coordinates (Step-1 for 0-based indexing)
                           cmap=discrete_cmap, s=40, alpha=0.8, vmin=0, vmax=12)  # Use discrete colormap with 13 distinct colors, no edge colors
ax1.set_xlabel('X Axis')  # Label the X axis
ax1.set_ylabel('Y Axis')  # Label the Y axis
ax1.set_zlabel('Z Axis')  # Label the Z axis
ax1.set_title('3D Scatter Plot of Coordinate Data by Step')  # Set plot title
cbar = fig1.colorbar(scatter_plot, ax=ax1, shrink=0.5, label='Step Number', ticks=range(13))  # Add color bar with discrete ticks
cbar.set_ticklabels(range(1, 14))  # Set tick labels to show steps 1-13

# Histogram showing step distribution
fig2, ax2 = plt.subplots(figsize=(9, 5))  # Create new figure and axis with 9x5 inch size
step_counts = df['Step'].value_counts().sort_index()  # Count occurrences of each step and sort by step number
ax2.bar(step_counts.index, step_counts.values, color='teal', edgecolor='black', width=0.7)  # Create bar chart with teal bars and black edges
ax2.set_xlabel('Step Number')  # Label the X axis
ax2.set_ylabel('Count')  # Label the Y axis
ax2.set_title('Distribution of Data Points Across Steps')  # Set plot title
ax2.set_xticks(range(1, 14))  # Set X-axis tick marks from 1 to 13
ax2.grid(axis='y', linestyle='--', alpha=0.5)  # Add horizontal grid lines with dashed style

# 2D projections - three separate plots
figure, subplots = plt.subplots(1, 3, figsize=(16, 5))  # Create figure with 1 row and 3 columns of subplots

# Plot 1: X vs Y
plot1 = subplots[0].scatter(df['X'], df['Y'], c=df['Step']-1,  # Create scatter plot of X vs Y coordinates (Step-1 for 0-based indexing)
                            cmap=discrete_cmap, s=25, alpha=0.7, vmin=0, vmax=12)  # Use discrete colormap without edge colors
subplots[0].set_xlabel('X Coordinate')  # Label X axis for first subplot
subplots[0].set_ylabel('Y Coordinate')  # Label Y axis for first subplot
subplots[0].set_title('2D View: X vs Y')  # Set title for first subplot
subplots[0].grid(True, linestyle=':', alpha=0.4)  # Add dotted grid lines
cbar1 = figure.colorbar(plot1, ax=subplots[0], ticks=range(13))  # Add color bar with discrete ticks
cbar1.set_ticklabels(range(1, 14))  # Set tick labels to show steps 1-13

# Plot 2: X vs Z
plot2 = subplots[1].scatter(df['X'], df['Z'], c=df['Step']-1,  # Create scatter plot of X vs Z coordinates (Step-1 for 0-based indexing)
                            cmap=discrete_cmap, s=25, alpha=0.7, vmin=0, vmax=12)  # Use discrete colormap without edge colors
subplots[1].set_xlabel('X Coordinate')  # Label X axis for second subplot
subplots[1].set_ylabel('Z Coordinate')  # Label Y axis for second subplot
subplots[1].set_title('2D View: X vs Z')  # Set title for second subplot
subplots[1].grid(True, linestyle=':', alpha=0.4)  # Add dotted grid lines
cbar2 = figure.colorbar(plot2, ax=subplots[1], ticks=range(13))  # Add color bar with discrete ticks
cbar2.set_ticklabels(range(1, 14))  # Set tick labels to show steps 1-13

# Plot 3: Y vs Z
plot3 = subplots[2].scatter(df['Y'], df['Z'], c=df['Step']-1,  # Create scatter plot of Y vs Z coordinates (Step-1 for 0-based indexing)
                            cmap=discrete_cmap, s=25, alpha=0.7, vmin=0, vmax=12)  # Use discrete colormap without edge colors
subplots[2].set_xlabel('Y Coordinate')  # Label X axis for third subplot
subplots[2].set_ylabel('Z Coordinate')  # Label Y axis for third subplot
subplots[2].set_title('2D View: Y vs Z')  # Set title for third subplot
subplots[2].grid(True, linestyle=':', alpha=0.4)  # Add dotted grid lines
cbar3 = figure.colorbar(plot3, ax=subplots[2], ticks=range(13))  # Add color bar with discrete ticks
cbar3.set_ticklabels(range(1, 14))  # Set tick labels to show steps 1-13

# Show all figures
plt.show()  # Display all created plots on screen




# Step 3: Correlation Analysis
######################################################
print("\n" + "-"*50)  # Print separator line
print("Step 3: Correlation Analysis")  # Print section title
print("-"*50)  # Print separator line

import seaborn as sns  # Import seaborn for correlation visualization
import numpy as np  # Import numpy for numerical operations

# Calculate Pearson correlation matrix
correlation_matrix = df.corr(method='pearson')  # Calculate Pearson correlation coefficients
print("\nPearson Correlation Matrix:")  # Print header
print(correlation_matrix)  # Display correlation matrix

# Create correlation heatmap
fig3, ax3 = plt.subplots(figsize=(8, 6))  # Create new figure with 8x6 inch size
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',  # Create heatmap with annotations using discrete-like colormap
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)  # Center at 0, square cells, with gridlines
ax3.set_title('Pearson Correlation Heatmap')  # Set plot title
plt.tight_layout()  # Adjust layout to prevent label cutoff

# Display correlation of features with target variable (Step)
print("\nCorrelation of Features with Target Variable (Step):")  # Print header
print(correlation_matrix['Step'].sort_values(ascending=False))  # Show correlations sorted by magnitude

plt.show()  # Display correlation heatmap
