import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv("ratings.csv")
print("Original Data")
print(df.head())

df.columns = ['userId','movieId','rating','timestamp']
df = df[df['userId'] != 'userId']
# DATA CLEANING

df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Remove missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()
df['label'] = df['rating'].apply(lambda x: 1 if x >= 3.5 else 0)

# Movie popularity
movie_popularity = df.groupby('movieId')['rating'].count()
df['movie_popularity'] = df['movieId'].map(movie_popularity)

# User average rating
user_avg = df.groupby('userId')['rating'].mean()
df['user_avg_rating'] = df['userId'].map(user_avg)
# ENCODING
le_user = LabelEncoder()
le_movie = LabelEncoder()

df['userId'] = le_user.fit_transform(df['userId'])
df['movieId'] = le_movie.fit_transform(df['movieId'])
print("\nPreprocessed Data:")
print(df.head(10))
# FEATURE SELECTION
X = df[['userId','movieId','movie_popularity','user_avg_rating']]
y = df['label']
# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
# SVM
svm = SVC(probability=True)
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, pred_lr))
print(classification_report(y_test, pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_lr))

print("\n--- KNN ---")
print("Accuracy:", accuracy_score(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print("\n--- SVM ---")
print("Accuracy:", accuracy_score(y_test, pred_svm))
print(classification_report(y_test, pred_svm))
sample = [[5, 120, 50, 3.5]] 
result = lr.predict(sample)
if result[0] == 1:
    print("\nRecommended Movie")
else:
    print("\nNot Recommended Movie")
prob = lr.predict_proba(X_test)[:,1]
fpr, tpr, threshold = roc_curve(y_test, prob)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Logistic Regression)")
plt.show()
