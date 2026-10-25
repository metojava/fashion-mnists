import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load the data
fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 2. Preprocess the data
# Scale pixel values to a 0-1 range and expand dimensions for the CNN channel
X_train = np.expand_dims(X_train, -1) / 255.0
X_test = np.expand_dims(X_test, -1) / 255.0

# Define clear human-readable class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 3. Build the CNN Model Architecture
modela = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Second Convolutional Block
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flattening and Dense Classification Layers
    layers.Flatten(),
    layers.Dropout(0.5), # Helps prevent overfitting
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax") # 10 outputs corresponding to classes
])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# 4. Compile the Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Train the Model
print("--- Training the model ---")
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# 6. Evaluate Performance on Test Data
print("\n--- Evaluating on unseen test data ---")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")

# 7. Make a Sample Prediction
predictions = model.predict(X_test[:1])
predicted_class = np.argmax(predictions[0])
print(f"Predicted class for the first test image: {class_names[predicted_class]}")
print(f"Actual class: {class_names[y_test[0]]}")
