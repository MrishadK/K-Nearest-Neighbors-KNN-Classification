# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for better visualization
iris_df = pd.DataFrame(X, columns=feature_names)
iris_df['target'] = y
iris_df['species'] = iris_df['target'].map({0: target_names[0], 1: target_names[1], 2: target_names[2]})

# Display dataset info
print("=== Dataset Information ===")
print(f"Features: {feature_names}")
print(f"Target classes: {target_names}")
print("\nFirst 5 rows:")
print(iris_df.head())

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Function to train and evaluate KNN
def evaluate_knn(k_values, X_train, y_train, X_test, y_test):
    results = []
    for k in k_values:
        # Create and train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Make predictions
        y_pred = knn.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'k': k,
            'accuracy': accuracy,
            'confusion_matrix': cm
        })
        
        # Print results for each k
        print(f"\n=== K = {k} ===")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
    
    return results

# Evaluate different k values
k_values = [1, 3, 5, 7, 10, 15]
print("\n=== Evaluating Different K Values ===")
results = evaluate_knn(k_values, X_train, y_train, X_test, y_test)

# Find the best k based on accuracy
best_k = max(results, key=lambda x: x['accuracy'])
print(f"\nBest K: {best_k['k']} with accuracy: {best_k['accuracy']:.4f}")

# Visualize decision boundaries (using first two features for 2D visualization)
def plot_decision_boundary(k, X, y):
    # Take only first two features for visualization
    X_2d = X[:, :2]
    
    # Train a KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_2d, y)
    
    # Plot decision regions
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X_2d, y, clf=knn, legend=2)
    plt.title(f'KNN Decision Regions (k={k})')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.show()

# Plot for different k values
print("\n=== Plotting Decision Boundaries ===")
for k in [1, 3, 5, 10]:
    plot_decision_boundary(k, X_scaled, y)

# Plot accuracy vs k values
accuracies = [result['accuracy'] for result in results]
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, 'bo-')
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. K Value')
plt.xticks(k_values)
plt.grid(True)
plt.show()