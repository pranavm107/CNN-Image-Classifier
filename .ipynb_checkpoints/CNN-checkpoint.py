import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical # it helps to convert labels to one-hot encoded labels it is a common preprocessing step for classification tasks
from tensorflow.keras.models import Sequential # it is a model container that allows you to stack multiple layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # it is a collection of different types of layers
import matplotlib as plt
import numpy as np


#STEP 1

# load CIFAR10 dataset
(X_train, y_train), (X_val, y_val) = cifar10.load_data()

print(X_train)

# Normalize the image data 
#scale the data to [0, 1] range
X_train = X_train.astype('float32') / 255.0 # normalize the data
X_val = X_val.astype('float32') / 255.0

# Apply one hot encoding to labels
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)

# cheking the shape of the data 
print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)
#output (50000, 32, 32, 3) (50000, 32, 32, 3) (50000, 10) (50000, 10)

#STEP 2
# building a CNN model

#1 herw we initialize the model 
model = sequential()

#2 first convolutional layer , it is 2d convolutioal layer that lerans filter that captures spatial featues in the input images
model.add((Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3))))

#3 max pooling layer , it is a pooling layer that downsamples the input image by taking the maximum value in each 2x2 pixel window
model.add(MaxPooling2D((2, 2))) # it takes the maximum value in each 2x2 pixel window

#4 second convolutional layer , it is 2d convolutioal layer that lerans filter that captures spatial featues in the input images
model.add((Conv2D(64, (3, 3), acrtivation = 'relu')))
model.add(MaxPooling2D(2, 2)) # it takes the maximum value in each 2x2 pixel window

#5 third convolutional layer , it is 2d convolutioal layer that lerans filter that captures spatial featues in the input images
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

#6 Flatten Layer ,this layer flattens the 3D output from the convolutional layers into a 1D vector
model.add(Flatten())

#7 Fully Connected dense layer, this layer uses the fully connected technique to transform the 1D vector into a 2D vector
model.add(Dense(128, activation = ' relu'))

#8 Drop out layer, this layer randomly drops out some of the neurons in the previous layer
model.add(Dropout(0.5))

#9 output Layer, this layer is used to output the results of the model
model.add(Dense(10, activation = 'softmax'))

#10 Summary of the model Architecture
model.summary()

# STEP 3:
#Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] ) # it is a function that compiles the model
# adam is a popular optimizer that is used to train the model
# categorical_crossentropy is a loss function that is used to measure the performance of the model
# accuracy is a metric that is used to measure the performance of the model

# STEP 4:
#Train the model
history = model.fit(X_train, y_train, epochs = 10, batch_size = 32,
                    validation_data = (X_val, y_val))
# fit is a function that trains the model
# X_train is the input data
# y_train is the target data
# epochs is the number of times the model will be trained
# batch_size is the number of samples in each batch
# validation_data is the data that will be used to evaluate the model

# STEP 5:
# Evaluate the model and Visualize Results
# After training, we can evaliuate the models accuracy on the test data and plot the training history

#Evaluate the model on test data

val_loss, val_acc = model.evalutae(X_val, y_val)
print('Test Accuarcy:', val_acc)

# Visulaize the results
#Plot the training and validation accuracy

plt.figure(figsize=(10, 6)) # create a figure with size 10x6, figsize means figure size, figure size is the size of the figure in inches

# Plot accuracy
# it is the same as above but for accuracy

plt.subplot(1, 2, 1) # create a subplot with 1 row and 2 columns
plt.plot(history.histoy['accuracy'], label = 'Training Accuracy') # plot the training accuracy, label is the name of the line
plt.plot(history.hstory['val_accuracy'], label = 'Validation Accuracy') # plot the validation accuracy, label is the name of the line
plt.xlabel(('Epochs')) # set the x-axis label to 'Epochs'
plt.ylabel(('Accuracy')) # set the y-axis label to 'Accuracy'
plt.legend() # add a legend

# Plot loss
# it is the same as above but for loss

plt.subplot(1, 2, 1) # create a subplot with 1 row and 2 columns
plt.plot(history.history['Loss'], label = 'Training Loss') # plot the training loss, label is the name of the line
plt.plot(history.history[val_loss], label = 'Validation loss') # plot the validation loss, label is the name of the line
plt.xlabel(('Epochs')) # set the x-axis label to 'Epochs'
plt.ylabel(('Loss')) # set the y-axis label to 'Loss'
plt. legend() # add a legend

plt.show() # show the plot


# Make prediction
# pick a random image from the test set

random_idx = 34567 # pick a random index from the test set
random_image = X_val[random_idx] # pick a random image from the test set
 # make prediction

prediction = model.predict(np.expand_dims(random_image, axis = 0))  # make prediction,expand_dims is used to add a new axis to the array, axis = 0 means add a new axis at the beginning
predicted_class = np.argmax(prediction, axis = 1) # argmax is used to get the index of the maximum value in the array, axis = 1 means get the index of the maximum value in the second axis

# print the actual and predicted class
print(f"Actual claa: (np.argmax(y_test[random_idx]))") 
print(f"Predicted Class: {predicted_class[0]}")


# Visualize the image prediction

plt.imshow(random_image)
plt.title(f"Predictedd: {predicted_class[0]}")
plt.show()