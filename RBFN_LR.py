import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Load the dataset from CSV file
data = pd.read_csv('Landslide1.csv')

# Split the dataset into training and testing sets
X = data.drop('value', axis=1) # Features
y = data['value'] # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### RBFN

# Step 1: Perform K-means clustering on the training data
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, n_init=10)
kmeans.fit(X_train)
centers = kmeans.cluster_centers_

# Step 2: Calculate the distance between each sample and each cluster center
distances = cdist(X_train, centers)

# Step 3: Apply the RBFN to the training data
sigma = np.mean(distances) / np.sqrt(2*n_clusters)
phi_train = np.exp(-(distances ** 2) / (2 * sigma ** 2))
phi_train = np.insert(phi_train, 0, 1, axis=1) # Add a column of ones for bias

# Step 4: Train the output weights using least squares regression
output_weights = np.linalg.inv(phi_train.T @ phi_train) @ phi_train.T @ y_train

# Step 5: Use the RBFN to predict the target variable for the testing set
distances_test = cdist(X_test, centers)
phi_test = np.exp(-(distances_test ** 2) / (2 * sigma ** 2))
phi_test = np.insert(phi_test, 0, 1, axis=1)
y_pred_rbf = np.round(phi_test @ output_weights)

# Step 6: Evaluate the model's accuracy and generate a classification report
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
report_rbf = classification_report(y_test, y_pred_rbf, zero_division=1)

# Calculate AUC for RBFN
fpr_rbf, tpr_rbf, thresholds_rbf = roc_curve(y_test, y_pred_rbf)
auc_rbf = roc_auc_score(y_test, y_pred_rbf)

### Logistic Regression

# Create a logistic regression model and fit it on the training data
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict the target variable for the testing set
y_pred_lr = model.predict(X_test)

# Evaluate the model's accuracy and generate a classification report
accuracy_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr, zero_division=1)

# Calculate AUC for logistic regression
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_pred_lr)

# Output the results
print("RBFN Accuracy:", accuracy_rbf)
print("RBFN AUC:", auc_rbf)
print("Logistic Regression Accuracy:", accuracy_lr)
print("Logistic Regression AUC:", auc_lr)

# Plot AUC curves
plt.plot(fpr_rbf, tpr_rbf, label='RBFN (AUC = {:.2f})'.format(auc_rbf))
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = {:.2f})'.format(auc_lr))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()