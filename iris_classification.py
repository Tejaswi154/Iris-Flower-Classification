
# 1. Import libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 2. Load the Iris dataset
data = load_iris()
X = data.data       # features: sepal length, sepal width, petal length, petal width
y = data.target     # labels: 0=setosa, 1=versicolor, 2=virginica

# 3. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create the Random Forest model
model = RandomForestClassifier(n_estimators=50, random_state=42)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions on the test set
predictions = model.predict(X_test)

# 7. Calculate and print accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# 8. Predict a new sample flower
sample_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
predicted_class = data.target_names[model.predict(sample_flower)[0]]
print("Predicted class for sample flower:", predicted_class)

# 9. Optional: Print all test predictions vs actual
print("\nTest Predictions vs Actual:")
for pred, actual in zip(predictions, y_test):
    print(f"Predicted: {data.target_names[pred]}, Actual: {data.target_names[actual]}")
