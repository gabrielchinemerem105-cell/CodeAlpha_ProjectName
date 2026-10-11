# CodeAlpha_ProjectName
#Data Science Tasks &amp; Instructions — CodeAlpha

# IRIS FLOWER CLASSIFICATION
# Machine Learning Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
df = pd.read_csv("Iris.csv")

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nSpecies distribution:")
print(df["Species"].value_counts())

# 2. Prepare input features (X) and target (y)
features = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

X = df[features]
y = df["Species"].str.replace("Iris-", "", regex=False)

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Create and train the model
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

model.fit(X_train, y_train)

print("\nModel training completed!")

# 5. Make predictions on unseen test data
y_pred = model.predict(X_test)

# 6. Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2%}")

# 7. Display precision, recall and F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Display confusion matrix
labels = model.named_steps[
    "kneighborsclassifier"
].classes_

cm = confusion_matrix(y_test, y_pred, labels=labels)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Iris Species Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.tight_layout()
plt.show()

# 9. Predict the species of a new flower
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=features
)

prediction = model.predict(new_flower)

print("\nNew flower measurements:", new_flower.values[0])
print("Predicted species:", prediction[0])
