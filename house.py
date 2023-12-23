import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Load the house price dataset
data = pd.read_csv("house_prices.csv")

# Separate features (X) and target variable (y)
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Option 1: Using scikit-learn LinearRegression**

# Create a linear regression model
lr_model = LinearRegression()

# Train the model on the training set
lr_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_lr = lr_model.predict(X_test)

# Evaluate model performance using mean squared error
lr_mse = mean_squared_error(y_test, y_pred_lr)
print("Linear Regression MSE:", lr_mse)

# **Option 2: Using TensorFlow for a simple neural network**

# Define the model structure
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer with a single neuron for house price prediction
])

# Compile the model with appropriate optimizer and loss function
model.compile(optimizer='adam', loss='mse')

# Train the model on the training set
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions on the testing set
y_pred_tf = model.predict(X_test)

# Evaluate model performance using mean squared error
tf_mse = mean_squared_error(y_test, y_pred_tf)
print("TensorFlow MSE:", tf_mse)

# **Remember:**
# - Adjust hyperparameters (e.g., epochs, batch_size) for better performance.
# - Explore different model architectures (e.g., deeper neural networks) with TensorFlow.
# - Use regularization techniques to prevent overfitting.
# - Evaluate model performance using various metrics (e.g., R-squared, MAE).
