import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from scipy.stats import norm

# Load Iris data
iris = datasets.load_iris()
X = iris.data
Y = iris.target

def CBN(X, Y):
    unique_classes = np.unique(Y)
    class_barycenters = {cls: np.mean(X[Y == cls], axis=0) for cls in unique_classes}
    class_priors = {cls: len(Y[Y == cls]) / len(Y) for cls in unique_classes}
    
    predictions = []
    for test_idx in range(len(X)):
        test_point = X[test_idx]
        probabilities = {}
        
        for cls in unique_classes:
            distances = np.linalg.norm(test_point - class_barycenters[cls])
            probabilities[cls] = class_priors[cls] / distances

        predicted_class = max(probabilities, key=probabilities.get)
        predictions.append(predicted_class)
    
    return predictions

def calculate_error(predictions, true_labels):
    return np.mean(predictions != true_labels) * 100

custom_predictions = CBN(X, Y)
custom_error = calculate_error(custom_predictions, Y)
print(f"Prediction error of custom CBN: {custom_error}%")


gnb = GaussianNB()
gnb.fit(X, Y)
gaussian_predictions = gnb.predict(X)
gaussian_error = calculate_error(gaussian_predictions, Y)
print(f"Prediction error of sklearn GaussianNB: {gaussian_error}%")
