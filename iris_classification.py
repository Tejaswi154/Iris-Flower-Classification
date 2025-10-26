# Iris Flower Classification using Random Forest with Plot
# Author: Tejaswi154
# Date: [Today]

# 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 2. Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create the Random Forest model
model = RandomForestClassifier(n_estimators=50, random_state=42)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
predictions = model.predict(X_test)

# 7. Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# 8. Predict a new sample flower
sample_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
predicted_class = data.target_names[model.predict(sample_flower)[0]]
print("Predicted class for sample flower:", predicted_class)

# 9. Plot Predicted vs Actual labels
plt.figure(figsize=(8,5))
plt.plot(range(len(y_test)), y_test, 'bo-', label='Actual')
plt.plot(range(len(predictions)), predictions, 'ro-', label='Predicted')
plt.title('Predicted vs Actual Iris Flower Labels')
plt.xlabel('Test Sample Index')
plt.ylabel('Flower Class (0=Setosa, 1=Versicolor, 2=Virginica)')
plt.legend()
plt.grid(True)
plt.show()
