import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

df = pd.read_csv('latestdata.csv', low_memory=False)
print(df.info())
print(df.describe())
missing_values = df.isnull().sum()
print(missing_values)

# Convert categorical variables to numerical
df['outcome'] = df['outcome'].apply(lambda x: 1 if x == 'died' else (0 if x == 'discharged' else np.nan))
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else (0 if x == 'female' else np.nan))
df['lives_in_Wuhan'] = df['lives_in_Wuhan'].apply(lambda x: 1 if x == 'yes' else (0 if x == 'no' else np.nan))


df = df.drop(columns=['travel_history_binary'])
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()
print(correlations)
correlation_with_target = correlations['outcome'].sort_values(ascending=False)
print(correlation_with_target)


plt.figure(figsize=(12, 8))
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
df = df.dropna(subset=['outcome'])
numeric_df = df[numerical_cols]
numeric_df = numeric_df.dropna()
print("Number of rows after dropping NaNs:", len(numeric_df))

# Check if numeric_df is not empty before applying PCA
if not numeric_df.empty:
    # PCA Analysis
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(numeric_df)

    # Scatter plot for PCA results
    plt.figure(figsize=(10, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=df['outcome'], cmap='viridis', alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of COVID-19 Dataset')
    plt.colorbar()
    plt.show()
else:
    print("The numeric dataframe is empty after handling missing values. PCA cannot be applied.")
