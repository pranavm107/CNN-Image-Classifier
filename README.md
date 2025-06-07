CNN Image Classification using CIFAR-10 Dataset
This project demonstrates how to build and train a Convolutional Neural Network (CNN) for image classification using the popular CIFAR-10 dataset. It includes model building, evaluation, performance visualization, and sample predictions — ideal for internship or academic projects.

📁 Project Structure
bash
Copy
Edit
CNN_Image_Classification/
│
├── CNN_Image_Classification_Report.pdf  # Final internship report
├── output/                        # Folder with output charts/images (optional)
└── README.md                      # This file
🧰 Technologies Used
Python 3

TensorFlow / Keras

NumPy

Matplotlib

CIFAR-10 Dataset

📊 Dataset Info
CIFAR-10 consists of 60,000 32x32 color images in 10 classes:

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

Split into:

50,000 training images

10,000 test images

🚀 How the Model Works
Load and preprocess CIFAR-10 data (normalization + one-hot encoding)

Build CNN with layers:

Conv2D → ReLU → MaxPooling

Conv2D → ReLU → MaxPooling

Flatten → Dense → Dropout

Compile using categorical_crossentropy and Adam optimizer

Train and validate

Visualize accuracy/loss graphs

Make sample predictions

📷 Sample Output
Training Accuracy	Sample Prediction

✅ How to Run the Project
Clone or download this repo

Open cnn_cifar10_model.ipynb in Jupyter/Colab/VSC

Run the cells in order

📄 Report
Detailed explanation of model building, results, and screenshots is available in:
📘 CNN_Image_Classification_Report.pdf

✍️ Author
Pranav M
Intern at Navodita Infotech
📧 pranavagneeshm@gmail.com

