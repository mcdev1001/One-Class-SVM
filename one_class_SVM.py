'''
About One Class SVM:
One-Class SVM is an unsupervised learning algorithm (SVM's are supervised)
primarily used for outlier detection. It learns normal data points (inliers) 
and then identifies instances that deviate from this representation as outliers.

outputs a binary decision (inlier or outlier: 1,-1) for new instances based 
on how close they are to the learned boundary. Instances lying in the boundary are 
considered inliers, while those outside are considered outliers.

One class SVM gets its name from the fact that it only requires one class 
to train data on (usually normal data).
'''


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from mlxtend.plotting import plot_decision_regions

# Read data 
df = pd.read_csv('/Users/devonmcdermott/Desktop/Streamlit_ui_capstone/injected dataset - injected dataset.csv')  

# Preprocessing
# Convert 'DATE' column to datetime format
df['DATE'] = pd.to_datetime(df['DATE'])

# Encode the 'AGENT' column using label encoding
label_encoder = LabelEncoder()
df['AGENT_encoded'] = label_encoder.fit_transform(df['AGENT'])

# Convert the date to numerical int values
df['DATE_numeric'] = df['DATE'].astype(np.int64)

# Feature selection
X = df[['DATE_numeric', 'AGENT_encoded']]

# Splitting data into train and test sets
X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)

# normalize/Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the One-Class SVM model
svm_model = OneClassSVM(kernel='rbf', nu=0.07)  # nu is the hyperparameter controlling the proportion of outliers
#the nu increases accuracy as it gets smaller?
svm_model.fit(X_train_scaled)

# Make predictions on the training data
train_predictions = svm_model.predict(X_train_scaled)

# Define a function to convert numerical date to date format
def numeric_to_date(numeric_date):
    return pd.to_datetime(numeric_date)
'''
# Visualizing the anomalies detected by the model in the training set
plt.figure(figsize=(10, 6))
plt.scatter(X_train['AGENT_encoded'], X_train['DATE_numeric'], c=train_predictions, cmap='viridis')
plt.xlabel('Agent')
plt.ylabel('Date')
plt.title('Anomalies Detected by One-Class SVM (Training Set)')
plt.colorbar(label='Anomaly Score')

# Get agent names and their corresponding encoded values
agent_encoded_names = dict(zip(df['AGENT_encoded'], df['AGENT']))

# Set x-axis ticks with agent names
plt.xticks(list(agent_encoded_names.keys()), list(agent_encoded_names.values()), rotation='vertical')

# Convert numerical scaled dates back to date format for y-axis ticks
plt.yticks(plt.yticks()[0], [numeric_to_date(int(tick)) for tick in plt.yticks()[0]])

plt.show()
'''






#TESTING
# Make predictions on the testing data
test_predictions = svm_model.predict(X_test_scaled)

# Visualizing the anomalies detected by the model in the testing set
plt.figure(figsize=(10, 6))
plt.scatter(X_test['AGENT_encoded'], X_test['DATE_numeric'], c=test_predictions, cmap='viridis')
plt.xlabel('Agent')
plt.ylabel('Date')
plt.title('Anomalies Detected by One-Class SVM (Testing Set)')
plt.colorbar(label='Anomaly Score')

# Get agent names and their corresponding encoded values
agent_encoded_names = dict(zip(df['AGENT_encoded'], df['AGENT']))

# Set x-axis ticks with actual agent names
plt.xticks(list(agent_encoded_names.keys()), list(agent_encoded_names.values()), rotation='vertical')

# Convert numerical scaled dates back to date format for y-axis ticks
plt.yticks(plt.yticks()[0], [numeric_to_date(int(tick)) for tick in plt.yticks()[0]])

plt.show()

#SVM PLOT
# Visualize decision boundary
plt.figure(figsize=(10, 6))
plot_decision_regions(X_test_scaled, test_predictions, clf=svm_model, legend=2)
plt.xlabel('Agent')
plt.ylabel('Date')
plt.title('One_Class_SVM Decision Boundary View')
plt.show()



#METRICS

# Calculate metrics for the training data
print("Metrics for the training data:")

#accuracy measures correctly classified instances/all instances
print("Accuracy: ", metrics.accuracy_score(train_predictions, [1] * len(train_predictions)))  # Assuming all predictions are normal (1)

#Precision measures true positive predictions/all positive predictions
# TP/TP+FP
print("Precision: ", metrics.precision_score(train_predictions, [1] * len(train_predictions)))

#the proportion of true positive predictions out of all actual positive instances in the dataset
#TP/TP+FN
print("Sensitivity: ", metrics.recall_score(train_predictions, [1] * len(train_predictions)))

# 2 * precision * sensitivity/precision + sensitivity
print("F1-Score: ", metrics.f1_score(train_predictions, [1] * len(train_predictions)))

#area under the ROC curve
#print("AUC: ", metrics.roc_auc_score(train_predictions, [1] * len(train_predictions)))



# Calculate metrics for the testing data
print("\nMetrics for the testing data:")
print("Accuracy: ", metrics.accuracy_score(test_predictions, [1] * len(test_predictions)))  # Assuming all predictions are normal (1)
print("Precision: ", metrics.precision_score(test_predictions, [1] * len(test_predictions)))
print("Sensitivity: ", metrics.recall_score(test_predictions, [1] * len(test_predictions)))
print("F1-Score: ", metrics.f1_score(test_predictions, [1] * len(test_predictions)))
#print("AUC: ", metrics.roc_auc_score(test_predictions, [1] * len(test_predictions)))

# Combine predictions from training and testing datasets
all_predictions = np.concatenate((train_predictions, test_predictions))

# Add the combined predictions as a new column to the DataFrame
df['svm_results'] = all_predictions

# Save the DataFrame to a new CSV file
df.to_csv('capstone_svm.csv', index=False)

