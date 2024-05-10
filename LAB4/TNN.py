import numpy as np
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

# Load Iris data
iris = datasets.load_iris()
X = iris.data
Y = iris.target

def TNN(X, Y):
    predictions = []
    for i in range(len(X)):

        train_X = np.delete(X, i, axis=0)
        train_Y = np.delete(Y, i)
        test_X = X[i]
        

        distances = pairwise_distances(test_X.reshape(1, -1), train_X).flatten()
        nearest_neighbor_idx = np.argmin(distances)
        predictions.append(train_Y[nearest_neighbor_idx])
        
    return predictions

def calculate_error(predictions, true_labels):
    return np.mean(predictions != true_labels) * 100


k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9 ,10]

results = {}
for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, Y)  
    predictions = knn.predict(X)
    error = calculate_error(predictions, Y)
    results[k] = error
    print(f"Prediction error for K={k}: {error}%")


best_k = min(results, key=results.get)
print(f"Best K value: {best_k} with the lowest error: {results[best_k]}%")
