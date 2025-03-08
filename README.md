
# Face Mask Detection using CNN

This project builds a **Convolutional Neural Network (CNN)** to detect whether a person is wearing a face mask or not. It utilizes a dataset from Kaggle and is trained using TensorFlow/Keras.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Prediction System](#prediction-system)
- [Technologies Used](#technologies-used)


---

## Installation

Follow these steps to set up and run the project:

1. Clone this repository:
   ```bash
   https://github.com/jowin-henry/Face_Mask_Detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle:
   ```bash
   kaggle datasets download -d omkargurav/face-mask-dataset
   ```

4. Extract the dataset:
   ```python
   from zipfile import ZipFile
   dataset = '/content/face-mask-dataset.zip'
   with ZipFile(dataset, 'r') as zip_ref:
       zip_ref.extractall()
       print('Dataset extracted successfully.')
   ```

---

## Dataset

The dataset contains two categories of images:

- **With Mask** (Label: `1`)
- **Without Mask** (Label: `0`)

### Dataset Statistics:
- **With Mask Images**: 3725
- **Without Mask Images**: 3828

---

## Model Architecture

A **Convolutional Neural Network (CNN)** is built using **TensorFlow/Keras** with the following layers:

1. **Convolutional Layer** (32 filters, kernel size 3x3, ReLU activation)
2. **MaxPooling Layer** (2x2)
3. **Convolutional Layer** (64 filters, kernel size 3x3, ReLU activation)
4. **MaxPooling Layer** (2x2)
5. **Flatten Layer**
6. **Fully Connected Layer** (128 neurons, ReLU activation, dropout)
7. **Fully Connected Layer** (64 neurons, ReLU activation, dropout)
8. **Output Layer** (2 neurons, sigmoid activation)

```python
import tensorflow as tf
from tensorflow import keras

num_of_classes = 2

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_of_classes, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
```

---

## Training the Model

The dataset is split into **training (80%)** and **testing (20%)** sets. The images are resized to `128x128` pixels and normalized.

```python
from sklearn.model_selection import train_test_split

# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Normalize pixel values
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

# Train the model
history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=10)
```

---

## Model Evaluation

The trained model is evaluated on the test set.

```python
loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test Accuracy:', accuracy)
```

### Training & Validation Loss Plot:
```python
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

### Training & Validation Accuracy Plot:
```python
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.legend()
plt.show()
```

---

## Prediction System

A predictive system allows users to input an image and determine whether the person is wearing a mask.

```python
import cv2
from google.colab.patches import cv2_imshow

input_image_path = input('Enter the path of the image: ')
input_image = cv2.imread(input_image_path)
cv2_imshow(input_image)

# Resize and normalize the image
input_image_resized = cv2.resize(input_image, (128,128))
input_image_scaled = input_image_resized / 255
input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])

# Make prediction
input_prediction = model.predict(input_image_reshaped)
input_pred_label = np.argmax(input_prediction)

# Display result
if input_pred_label == 1:
    print('The person is wearing a mask 😷')
else:
    print('The person is NOT wearing a mask ❌')
```

---

## Technologies Used

- **Python**
- **TensorFlow/Keras** (for Deep Learning)
- **OpenCV** (for Image Processing)
- **Pandas & NumPy** (for Data Handling)
- **Matplotlib** (for Visualization)
- **scikit-learn** (for Data Splitting)

---


