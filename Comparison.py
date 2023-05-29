import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
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

# Get the selected feature names
selected_feature_names = X.columns[selector.get_support()].tolist()
#print("Selected Features:", selected_feature_names)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict with Random Forest Classifier
y_pred_rf = rf_classifier.predict(X_test)

# Calculate accuracy for Random Forest Classifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# Calculate predicted probabilities for Random Forest Classifier
y_pred_proba_rf = rf_classifier.predict_proba(X_test)[:, 1]

# Calculate AUC-ROC score for Random Forest Classifier
auc_roc_rf = roc_auc_score(y_test, y_pred_proba_rf)
print("Random Forest AUC-ROC:", auc_roc_rf)

# Plot ROC curve for Random Forest Classifier
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_proba_rf)
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = {:.2f})'.format(auc_roc_rf))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# XGBoost Classifier
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train, y_train)

# Predict with XGBoost Classifier
y_pred_xgb = xgb_classifier.predict(X_test)

# Calculate accuracy for XGBoost Classifier
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy:", accuracy_xgb)

# Calculate predicted probabilities for XGBoost Classifier
y_pred_proba_xgb = xgb_classifier.predict_proba(X_test)[:, 1]

# Calculate AUC-ROC score for XGBoost Classifier
auc_roc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
print("XGBoost AUC-ROC:", auc_roc_xgb)

# Plot ROC curve for XGBoost Classifier
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_pred_proba_xgb)
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost (AUC = {:.2f})'.format(auc_roc_xgb))

# SVM Classifier
svm_classifier = SVC(probability=True, random_state=42)
svm_classifier.fit(X_train, y_train)

# Predict with SVM Classifier
y_pred_svm = svm_classifier.predict(X_test)

# Calculate accuracy for SVM Classifier
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Calculate predicted probabilities for SVM Classifier
y_pred_proba_svm = svm_classifier.predict_proba(X_test)[:, 1]

# Calculate AUC-ROC score for SVM Classifier
auc_roc_svm = roc_auc_score(y_test, y_pred_proba_svm)
print("SVM AUC-ROC:", auc_roc_svm)

# Plot ROC curve for SVM Classifier
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_pred_proba_svm)
plt.plot(fpr_svm, tpr_svm, label='SVM (AUC = {:.2f})'.format(auc_roc_svm))

plt.legend()
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

# Print confusion matrix for each classifier
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
