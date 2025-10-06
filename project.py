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





# Step 4: Classification Model Development/Engineering
######################################################
print("\n" + "-"*50)  # Print separator line
print("Step 4: Classification Model Development")  # Print section title
print("-"*50)  # Print separator line

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV  # Import model selection tools
from sklearn.svm import SVC  # Import Support Vector Machine classifier
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression classifier
from sklearn.preprocessing import StandardScaler  # Import scaler for feature normalization
from sklearn.pipeline import Pipeline  # Import Pipeline for combining preprocessing and modeling
from scipy.stats import uniform  # Import distributions for RandomizedSearchCV

# Separate features and target
features = df[['X', 'Y', 'Z']]  # Extract coordinate features
target = df['Step']  # Extract maintenance step target

# Create train-test split with 80-20 ratio
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target)  # Stratified split to maintain class balance

print(f"\nDataset split completed:")  # Print header
print(f"  Training samples: {len(features_train)}")  # Display training count
print(f"  Testing samples: {len(features_test)}")  # Display testing count
print(f"  Features: {features_train.shape[1]}")  # Display feature count
print(f"  Target classes: {target.nunique()}")  # Display number of classes

# Model 1: Logistic Regression with GridSearchCV using Pipeline
print("\n" + "="*60)  # Print separator
print("Model 1: Logistic Regression (GridSearchCV + Pipeline)")  # Print model header
print("="*60)  # Print separator

# Build pipeline with scaling and logistic regression
logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize features
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))  # Step 2: Logistic Regression
])

# Hyperparameter grid with pipeline naming convention
logistic_params = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
    'classifier__solver': ['newton-cg', 'lbfgs', 'saga'],  # Solvers for optimization
    'classifier__penalty': ['l2']  # Penalty types
}

# Execute grid search
logistic_grid = GridSearchCV(logistic_pipeline, logistic_params, cv=5,
                            scoring='accuracy', n_jobs=-1, verbose=1)  # 5-fold cross-validation
print("Initiating Logistic Regression training...")  # Status message
logistic_grid.fit(features_train, target_train)  # Train model

# Report results
print(f"\nOptimal parameters: {logistic_grid.best_params_}")  # Best hyperparameters found
print(f"Cross-validation accuracy: {logistic_grid.best_score_:.4f}")  # CV accuracy
print(f"Test set accuracy: {logistic_grid.score(features_test, target_test):.4f}")  # Test accuracy

# Model 2: Support Vector Classifier with GridSearchCV using Pipeline
print("\n" + "="*60)  # Print separator
print("Model 2: Support Vector Classifier (GridSearchCV + Pipeline)")  # Print model header
print("="*60)  # Print separator

# Build SVC pipeline
svc_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Feature scaling
    ('classifier', SVC(random_state=42))  # Step 2: SVC
])

# Hyperparameter grid for SVC
svc_params = {
    'classifier__C': [0.1, 1, 10, 50],  # Regularization parameter
    'classifier__kernel': ['linear', 'rbf', 'sigmoid'],  # Kernel functions
    'classifier__gamma': ['scale', 'auto', 0.01]  # Kernel coefficient
}

# Execute grid search
svc_grid = GridSearchCV(svc_pipeline, svc_params, cv=5,
                       scoring='accuracy', n_jobs=-1, verbose=1)  # 5-fold CV
print("Initiating SVC training...")  # Status message
svc_grid.fit(features_train, target_train)  # Train model

# Report results
print(f"\nOptimal parameters: {svc_grid.best_params_}")  # Best parameters
print(f"Cross-validation accuracy: {svc_grid.best_score_:.4f}")  # CV score
print(f"Test set accuracy: {svc_grid.score(features_test, target_test):.4f}")  # Test score

# Model 3: Decision Tree with GridSearchCV using Pipeline
print("\n" + "="*60)  # Print separator
print("Model 3: Decision Tree (GridSearchCV + Pipeline)")  # Print model header
print("="*60)  # Print separator

from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree classifier

# Build Decision Tree pipeline (scaling included for consistency)
dt_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardization
    ('classifier', DecisionTreeClassifier(random_state=42))  # Step 2: Decision Tree Classifier
])

# Hyperparameter grid for Decision Tree
dt_params = {
    'classifier__max_depth': [5, 10, 15, 20, None],  # Tree depth
    'classifier__min_samples_split': [2, 5, 10, 15],  # Split threshold
    'classifier__min_samples_leaf': [1, 2, 4, 6],  # Leaf size threshold
    'classifier__criterion': ['gini', 'entropy'],  # Splitting criterion
    'classifier__splitter': ['best', 'random']  # Splitting strategy
}

# Execute grid search
dt_grid = GridSearchCV(dt_pipeline, dt_params, cv=5,
                      scoring='accuracy', n_jobs=-1, verbose=1)  # 5-fold CV
print("Initiating Decision Tree training...")  # Status message
dt_grid.fit(features_train, target_train)  # Train model

# Report results
print(f"\nOptimal parameters: {dt_grid.best_params_}")  # Best parameters
print(f"Cross-validation accuracy: {dt_grid.best_score_:.4f}")  # CV score
print(f"Test set accuracy: {dt_grid.score(features_test, target_test):.4f}")  # Test score

# Model 4: Support Vector Classifier with RandomizedSearchCV using Pipeline
print("\n" + "="*60)  # Print separator
print("Model 4: Support Vector Classifier (RandomizedSearchCV + Pipeline)")  # Print model header
print("="*60)  # Print separator

# Build SVC pipeline for randomized search
svc_random_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Feature scaling
    ('classifier', SVC(random_state=42))  # Step 2: SVC
])

# Define parameter distributions for random search
svc_random_params = {
    'classifier__C': uniform(0.1, 100),  # Continuous distribution between 0.1 and 100
    'classifier__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],  # Kernel functions
    'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]  # Kernel coefficient options
}

# Execute randomized search with 50 iterations
svc_random = RandomizedSearchCV(svc_random_pipeline, svc_random_params, n_iter=50,
                               cv=5, scoring='accuracy', n_jobs=-1, verbose=1,
                               random_state=42)  # 50 random parameter combinations
print("Initiating SVC with RandomizedSearchCV...")  # Status message
svc_random.fit(features_train, target_train)  # Train model

# Report results
print(f"\nOptimal parameters: {svc_random.best_params_}")  # Best parameters found
print(f"Cross-validation accuracy: {svc_random.best_score_:.4f}")  # CV score
print(f"Test set accuracy: {svc_random.score(features_test, target_test):.4f}")  # Test score

# Comparative summary table
print("\n" + "="*60)  # Print separator
print("COMPARATIVE ANALYSIS - All Models")  # Summary header
print("="*60)  # Print separator
print(f"\n{'Algorithm':<45} {'Test Accuracy':>12}")  # Table header
print("-"*60)  # Divider
print(f"{'Logistic Regression (Grid)':<45} {logistic_grid.score(features_test, target_test):>12.4f}")  # LR score
print(f"{'Support Vector Classifier (Grid)':<45} {svc_grid.score(features_test, target_test):>12.4f}")  # SVC score
print(f"{'Decision Tree (Grid)':<45} {dt_grid.score(features_test, target_test):>12.4f}")  # DT Grid score
print(f"{'Support Vector Classifier (Randomized)':<45} {svc_random.score(features_test, target_test):>12.4f}")  # SVC Random score
print("="*60)  # Print separator




# Step 5: Model Performance Analysis
######################################################
print("\n" + "-"*50)  # Print separator line
print("Step 5: Model Performance Analysis")  # Print section title
print("-"*50)  # Print separator line

from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, classification_report  # Import metrics

# Dictionary to store performance metrics for all models
model_results = {}

# Evaluate Model 1: Logistic Regression
print("\nEvaluating Logistic Regression...")  # Status message
lr_predictions = logistic_grid.predict(features_test)  # Generate predictions
model_results['Logistic Regression'] = {
    'predictions': lr_predictions,
    'accuracy': accuracy_score(target_test, lr_predictions),  # Calculate accuracy
    'precision': precision_score(target_test, lr_predictions, average='weighted'),  # Calculate weighted precision
    'f1_score': f1_score(target_test, lr_predictions, average='weighted')  # Calculate weighted F1-score
}

# Evaluate Model 2: Support Vector Classifier (Grid)
print("Evaluating Support Vector Classifier (Grid)...")  # Status message
svc_predictions = svc_grid.predict(features_test)  # Generate predictions
model_results['SVC (GridSearchCV)'] = {
    'predictions': svc_predictions,
    'accuracy': accuracy_score(target_test, svc_predictions),  # Calculate accuracy
    'precision': precision_score(target_test, svc_predictions, average='weighted'),  # Calculate weighted precision
    'f1_score': f1_score(target_test, svc_predictions, average='weighted')  # Calculate weighted F1-score
}

# Evaluate Model 3: Decision Tree
print("Evaluating Decision Tree...")  # Status message
dt_predictions = dt_grid.predict(features_test)  # Generate predictions
model_results['Decision Tree'] = {
    'predictions': dt_predictions,
    'accuracy': accuracy_score(target_test, dt_predictions),  # Calculate accuracy
    'precision': precision_score(target_test, dt_predictions, average='weighted'),  # Calculate weighted precision
    'f1_score': f1_score(target_test, dt_predictions, average='weighted')  # Calculate weighted F1-score
}

# Evaluate Model 4: Support Vector Classifier (Randomized)
print("Evaluating Support Vector Classifier (Randomized)...")  # Status message
svc_random_predictions = svc_random.predict(features_test)  # Generate predictions
model_results['SVC (RandomizedSearchCV)'] = {
    'predictions': svc_random_predictions,
    'accuracy': accuracy_score(target_test, svc_random_predictions),  # Calculate accuracy
    'precision': precision_score(target_test, svc_random_predictions, average='weighted'),  # Calculate weighted precision
    'f1_score': f1_score(target_test, svc_random_predictions, average='weighted')  # Calculate weighted F1-score
}

# Display performance comparison table
print("\n" + "="*80)  # Print separator
print("PERFORMANCE METRICS COMPARISON")  # Table header
print("="*80)  # Print separator
print(f"{'Model':<35} {'Accuracy':<15} {'Precision':<15} {'F1-Score':<15}")  # Column headers
print("-"*80)  # Divider

for model_name, metrics in model_results.items():
    print(f"{model_name:<35} {metrics['accuracy']:<15.4f} {metrics['precision']:<15.4f} {metrics['f1_score']:<15.4f}")  # Print metrics

print("="*80)  # Print separator

# Determine best performing model based on accuracy
best_model = max(model_results, key=lambda x: model_results[x]['accuracy'])  # Find model with highest accuracy
best_accuracy = model_results[best_model]['accuracy']  # Get best accuracy
best_predictions = model_results[best_model]['predictions']  # Get best model predictions

print(f"\nBest Performing Model: {best_model}")  # Display best model
print(f"  Accuracy:  {best_accuracy:.4f}")  # Display accuracy
print(f"  Precision: {model_results[best_model]['precision']:.4f}")  # Display precision
print(f"  F1-Score:  {model_results[best_model]['f1_score']:.4f}")  # Display F1-score

# Generate confusion matrix for best model
conf_matrix = confusion_matrix(target_test, best_predictions)  # Calculate confusion matrix

# Create confusion matrix visualization
plt.figure(figsize=(12, 10))  # Create figure
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',  # Create heatmap with annotations
            xticklabels=sorted(target_test.unique()),  # X-axis labels (predicted)
            yticklabels=sorted(target_test.unique()),  # Y-axis labels (actual)
            cbar_kws={'label': 'Count'})  # Colorbar label
plt.xlabel('Predicted Maintenance Step', fontsize=12)  # X-axis label
plt.ylabel('Actual Maintenance Step', fontsize=12)  # Y-axis label
plt.title(f'Confusion Matrix: {best_model}\nAccuracy: {best_accuracy:.4f}', fontsize=14, pad=20)  # Title
plt.tight_layout()  # Adjust layout

# Display detailed classification report
print("\n" + "="*80)  # Print separator
print(f"CLASSIFICATION REPORT: {best_model}")  # Report header
print("="*80)  # Print separator
print(classification_report(target_test, best_predictions,
                           target_names=[f'Step {i}' for i in sorted(target_test.unique())]))  # Print detailed report




# Step 6: Stacked Model Performance Analysis
######################################################
print("\n" + "-"*50)  # Print separator line
print("Step 6: Stacked Model Performance Analysis")  # Print section title
print("-"*50)  # Print separator line

from sklearn.ensemble import StackingClassifier  # Import StackingClassifier

# Create stacking ensemble combining SVC (Grid) and Decision Tree
print("\nBuilding Stacked Model (SVC + Decision Tree)...")  # Status message
stacking_model = StackingClassifier(
    estimators=[
        ('svc', svc_grid.best_estimator_),  # SVC from GridSearchCV
        ('decision_tree', dt_grid.best_estimator_)  # Decision Tree from GridSearchCV
    ],
    final_estimator=LogisticRegression(random_state=42, max_iter=1000),  # Meta-classifier
    cv=5  # 5-fold cross-validation for training meta-model
)

# Train stacked model
print("Training stacked model...")  # Status message
stacking_model.fit(features_train, target_train)  # Fit on training data

# Generate predictions
stacked_predictions = stacking_model.predict(features_test)  # Predict on test data

# Calculate performance metrics
stacked_accuracy = accuracy_score(target_test, stacked_predictions)  # Accuracy
stacked_precision = precision_score(target_test, stacked_predictions, average='weighted')  # Precision
stacked_f1 = f1_score(target_test, stacked_predictions, average='weighted')  # F1-score

# Display stacked model performance
print("\n" + "="*80)  # Print separator
print("STACKED MODEL PERFORMANCE")  # Header
print("="*80)  # Print separator
print(f"Accuracy:  {stacked_accuracy:.4f}")  # Display accuracy
print(f"Precision: {stacked_precision:.4f}")  # Display precision
print(f"F1-Score:  {stacked_f1:.4f}")  # Display F1-score
print("="*80)  # Print separator

# Compare with individual models
print(f"\nComparison with Base Models:")  # Comparison header
print(f"  SVC (GridSearchCV):     Accuracy = {model_results['SVC (GridSearchCV)']['accuracy']:.4f}")  # SVC accuracy
print(f"  Decision Tree:          Accuracy = {model_results['Decision Tree']['accuracy']:.4f}")  # DT accuracy
print(f"  Stacked Model:          Accuracy = {stacked_accuracy:.4f}")  # Stacked accuracy

# Calculate improvement or difference
accuracy_diff = stacked_accuracy - max(model_results['SVC (GridSearchCV)']['accuracy'],
                                        model_results['Decision Tree']['accuracy'])  # Difference
if accuracy_diff > 0.001:
    print(f"\nStacking improved accuracy by {accuracy_diff:.4f}")  # Improvement message
elif accuracy_diff < -0.001:
    print(f"\nStacking decreased accuracy by {abs(accuracy_diff):.4f}")  # Decrease message
else:
    print(f"\nStacking showed minimal change in accuracy ({accuracy_diff:.4f})")  # Minimal change message

# Generate confusion matrix for stacked model
stacked_conf_matrix = confusion_matrix(target_test, stacked_predictions)  # Calculate confusion matrix

# Create confusion matrix visualization
plt.figure(figsize=(12, 10))  # Create figure
sns.heatmap(stacked_conf_matrix, annot=True, fmt='d', cmap='Purples',  # Create heatmap with annotations
            xticklabels=sorted(target_test.unique()),  # X-axis labels
            yticklabels=sorted(target_test.unique()),  # Y-axis labels
            cbar_kws={'label': 'Count'})  # Colorbar label
plt.xlabel('Predicted Maintenance Step', fontsize=12)  # X-axis label
plt.ylabel('Actual Maintenance Step', fontsize=12)  # Y-axis label
plt.title(f'Confusion Matrix: Stacked Model (SVC + Decision Tree)\nAccuracy: {stacked_accuracy:.4f}',
          fontsize=14, pad=20)  # Title
plt.tight_layout()  # Adjust layout

# Display classification report for stacked model
print("\n" + "="*80)  # Print separator
print("CLASSIFICATION REPORT: Stacked Model")  # Report header
print("="*80)  # Print separator
print(classification_report(target_test, stacked_predictions,
                           target_names=[f'Step {i}' for i in sorted(target_test.unique())]))  # Print detailed report

# Display all plots at the end
plt.show(block=False)  # Show all plots without blocking
input("\nPress Enter to close all plots and exit...")  # Wait for user input
plt.close('all')  # Close all plot windows
