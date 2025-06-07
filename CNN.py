# Import necessary libraries
import tensorflow as tf  # TensorFlow is a deep learning library
from tensorflow.keras.datasets import cifar10  # CIFAR-10 is a dataset of 60,000 32x32 color images in 10 classes
from tensorflow.keras.utils import to_categorical  # Converts labels (like 3) into one-hot format (like [0,0,0,1,0,0,0,0,0,0])
from tensorflow.keras.models import Sequential  # Sequential is a linear stack of layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # CNN layers used for image processing
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Used to stop training early or reduce learning rate
from sklearn.model_selection import train_test_split  # Splits data into training and validation sets
import matplotlib.pyplot as plt  # Used to plot graphs
import numpy as np  # Used for numerical operations like random number generation

# STEP 1: Load CIFAR-10 dataset (already split into training and testing data)
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values: changes values from [0, 255] to [0, 1]
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert labels to one-hot encoded vectors
y_train_full = to_categorical(y_train_full, 10)  # 10 is the number of classes
y_test = to_categorical(y_test, 10)

# Split full training data into a smaller training set and a validation set (20% for validation)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# STEP 2: Build a Convolutional Neural Network (CNN)
model = Sequential()  # Start building a model layer by layer

# Add first convolutional layer
# Conv2D: 32 filters, each of size 3x3, activation ReLU, input image shape 32x32 with 3 channels (RGB)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

# Add first max pooling layer: takes max value from each 2x2 block (reduces image size)
model.add(MaxPooling2D((2, 2)))

# Add second convolutional layer with 64 filters
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add second max pooling layer
model.add(MaxPooling2D((2, 2)))

# Add third convolutional layer with 128 filters
model.add(Conv2D(128, (3, 3), activation='relu'))

# Add third max pooling layer
model.add(MaxPooling2D((2, 2)))

# Flatten layer: converts 2D data into 1D (required for dense layers)
model.add(Flatten())

# Add dense (fully connected) layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='relu'))

# Dropout layer: randomly turns off 50% neurons during training to prevent overfitting
model.add(Dropout(0.5))  # 0.5 means 50% dropout

# Final output layer: 10 neurons (for 10 classes), using softmax to predict probabilities
model.add(Dense(10, activation='softmax'))

# Print the full model architecture
model.summary()

# STEP 3: Compile the model
model.compile(
    optimizer='adam',  # Adam is an advanced optimizer that adjusts learning rate automatically
    loss='categorical_crossentropy',  # Used when labels are one-hot encoded
    metrics=['accuracy']  # We want to track accuracy
)

# Add callbacks
# EarlyStopping: stops training if validation loss doesn't improve for 3 epochs
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ReduceLROnPlateau: reduces the learning rate if validation loss plateaus for 2 epochs
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# STEP 4: Train the model
history = model.fit(
    X_train, y_train,  # Input images and labels
    epochs=20,  # Maximum number of times to train the full dataset
    batch_size=32,  # Number of images trained together in one step
    validation_data=(X_val, y_val),  # Validate on unseen data
    callbacks=[early_stop, reduce_lr]  # Use the callbacks defined above
)

# STEP 5: Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_acc)

# STEP 6: Plot training history

# figsize: defines the size of the plot window (12 inches wide, 5 inches tall)
plt.figure(figsize=(12, 5))

# Subplot 1: Accuracy
plt.subplot(1, 2, 1)  # Create a 1 row, 2 column grid, current plot is 1st
plt.plot(history.history['accuracy'], label='Training Accuracy')  # Plot training accuracy
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Plot validation accuracy
plt.xlabel('Epochs')  # Label for X-axis
plt.ylabel('Accuracy')  # Label for Y-axis
plt.title('Accuracy over Epochs')  # Title of the graph
plt.legend()  # Show labels

# Subplot 2: Loss
plt.subplot(1, 2, 2)  # 2nd plot
plt.plot(history.history['loss'], label='Training Loss')  # Plot training loss
plt.plot(history.history['val_loss'], label='Validation Loss')  # Plot validation loss
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.tight_layout()  # Adjust layout to prevent overlapping text
plt.show()  # Display the plots

# STEP 7: Make a prediction on a random image

# CIFAR-10 class names (for display)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Select a random image index from the test set
random_idx = np.random.randint(0, len(X_test))  # Generate a random number
random_image = X_test[random_idx]  # Get the image

# Predict class for this image
# np.expand_dims adds an extra dimension so that shape becomes (1, 32, 32, 3)
prediction = model.predict(np.expand_dims(random_image, axis=0))
predicted_class = np.argmax(prediction, axis=1)  # Find index of highest predicted probability

# Print actual and predicted class names
print(f"Actual Class: {class_names[np.argmax(y_test[random_idx])]}")
print(f"Predicted Class: {class_names[predicted_class[0]]}")

# Show the image with predicted label
plt.imshow(random_image)  # Show the image
plt.title(f"Predicted: {class_names[predicted_class[0]]}")  # Add prediction as title
plt.axis('off')  # Hide axis
plt.show()  # Display the image
