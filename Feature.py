#code for feature selection : 
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Load the dataset from CSV file
data = pd.read_csv('Landslide1.csv')

# Perform Feature Selection using Mutual Information
X = data.drop('value', axis=1) # Features
y = data['value'] # Target variable

# Calculate mutual information between each feature and the target variable
mutual_info_scores = mutual_info_classif(X, y)

# Create a dataframe to store the feature names and their scores
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': mutual_info_scores})

# Sort the features based on their scores in ascending order
feature_scores = feature_scores.sort_values(by='Score', ascending=False)

# Assign ranks to the features based on their scores
feature_scores['Rank'] = range(1, len(feature_scores) + 1)

# Output the features with their mutual information scores and rankings
print("Features Ranked by Mutual Information Score:")
for idx, row in feature_scores.iterrows():
    print("Rank {}: {} - Score: {}".format(row['Rank'], row['Feature'], row['Score']))
