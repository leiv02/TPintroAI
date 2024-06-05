import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_squared_error, silhouette_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import re

# Function to clean and convert age values
def clean_age(age):
    if pd.isnull(age) or age == '':
        return np.nan
    if isinstance(age, str):
        if '-' in age:
            parts = age.split('-')
            if parts[0].isdigit() and parts[1].isdigit():
                return (float(parts[0]) + float(parts[1])) / 2
            else:
                return np.nan
        else:
            match = re.findall(r'\d+', age)
            if match:
                return float(match[0])
            else:
                return np.nan
    return float(age)

# Load the dataset
df = pd.read_csv('latestdata.csv', low_memory=False)
df['outcome'] = df['outcome'].map({'died': 0, 'discharged': 1})
df.dropna(subset=df.select_dtypes(include=[np.number]).columns, inplace=True)

# Part 1: Predicting the Outcome using K-NN

# Separate features and target for outcome prediction
X_outcome = df.drop(columns=['outcome']).select_dtypes(include=[np.number])
y_outcome = df['outcome']

# Ensure there are no missing values in X_outcome and y_outcome
X_outcome = X_outcome.dropna()
y_outcome = y_outcome.loc[X_outcome.index]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_outcome, y_outcome, test_size=0.3, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Standardize the data
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test)

# Define the K-NN model and parameters for Grid-Search
knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Perform Grid-Search
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_smote, y_train_smote)

# Print the best parameters and the best score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

# Train the K-NN model with the best parameters
best_knn = grid_search.best_estimator_
best_knn.fit(X_train_smote, y_train_smote)

# Predict and evaluate
y_pred_outcome = best_knn.predict(X_test)
print("Confusion Matrix (Outcome Prediction):")
print(confusion_matrix(y_test, y_pred_outcome))
print("Accuracy Score (Outcome Prediction):")
print(accuracy_score(y_test, y_pred_outcome))
print("Classification Report (Outcome Prediction):")
print(classification_report(y_test, y_pred_outcome))

# Part 2: Regression to Predict Age

# Clean the 'age' column
df['age'] = df['age'].apply(clean_age)

# Ensure the 'age' column has no missing values
df_age = df[df['age'].notna()].copy()

# Select features and target for regression
X_reg = df_age.drop(columns=['age']).select_dtypes(include=[np.number])
y_reg = df_age['age']

# Ensure there are no missing values in X_reg and y_reg
X_reg = X_reg.dropna()
y_reg = y_reg.loc[X_reg.index]

# Split the data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Standardize the data
X_train_reg = scaler.fit_transform(X_train_reg)
X_test_reg = scaler.transform(X_test_reg)

# Train the regression model
reg = LinearRegression()
reg.fit(X_train_reg, y_train_reg)

# Predict and evaluate
y_pred_age = reg.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_age)
print(f"Mean Squared Error (Age Prediction): {mse}")

# Part 3: Clustering using K-means

# Prepare the data for clustering
X_clustering = df.select_dtypes(include=[np.number]).dropna()

# Standardize the data
X_clustering = scaler.fit_transform(X_clustering)

# Determine the best number of clusters using the silhouette score
silhouette_scores = []
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_clustering)
    silhouette_avg = silhouette_score(X_clustering, cluster_labels)
    silhouette_scores.append((n_clusters, silhouette_avg))

# Find the best number of clusters
best_n_clusters = max(silhouette_scores, key=lambda item: item[1])[0]
print(f"Best number of clusters: {best_n_clusters}")

# Apply K-means with the best number of clusters
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_clustering)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot([score[0] for score in silhouette_scores], [score[1] for score in silhouette_scores], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for K-means Clustering')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(X_clustering[:, 0], X_clustering[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.colorbar(label='Cluster')
plt.show()
