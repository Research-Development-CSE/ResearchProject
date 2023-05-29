import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset from CSV file
df = pd.read_csv('landslide.csv')

# Separate the target variable from features
X = df.drop('layer', axis=1)
y = df['layer']

# Shift the data to make it non-negative
X_min = X.min()
X = X - X_min
y = y.replace({'landslide_zone': 1, 'non_landslidezone': 0})

# Convert class labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Perform feature selection using chi-square test
selector = SelectKBest(chi2, k=5)  # Select the top 5 features
X_selected = selector.fit_transform(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict the test data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate predicted probabilities
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

# Calculate AUC-ROC score
auc_roc = roc_auc_score(y_test, y_pred_proba)
print("AUC-ROC:", auc_roc)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
