Your Guide to Convolutional Neural Networks (CNNs)

This guide will walk you through the fundamental concepts of CNNs, connecting the theory from your PDF and lecture images to a practical code example for classifying chest X-ray images.

---

### Part 1: The "Why" - Limitations of Regular Neural Networks for Images

In our first session, we used a Dense (fully-connected) network for MNIST digits. It worked, but it has huge problems with larger, real-world images:

* **Too Many Parameters:** A 200x200 RGB image has 120,000 inputs. With just 512 neurons, that’s \~61 million weights in one layer!
* **Loses Spatial Structure:** Flattening loses spatial relationships between pixels.

CNNs were invented to solve these exact problems.

---

### Part 2: The "How" - The Core Idea of CNNs

CNNs mimic the human visual system by detecting local patterns and building up complexity.

* **Sparse Connectivity:** A neuron connects only to a local patch of the image.
* **Parameter Sharing:** The same kernel scans the whole image to detect repeating features.

This greatly reduces parameters and improves generalization.

---

### Part 3: The Building Blocks of a CNN

#### 3.1. The Convolutional Layer and the Kernel

* A **kernel** is a small matrix (3x3 or 5x5) of weights.
* **Convolution** slides the kernel over the image, performing element-wise multiplication and sum, producing a **feature map**.
* A Conv2D layer with `filters=32` learns 32 such kernels, resulting in 32 feature maps.

#### 3.2. The Pooling Layer

* **Max Pooling** (most common): Takes the max value from a small window.
* Purpose: Reduce feature map size, make the model robust to small shifts, and reduce overfitting.

#### 3.3. The Fully-Connected (Dense) Layer

* After several Conv and Pool layers, we **Flatten** the feature maps.
* Dense layers interpret features and produce final predictions.

---

### Part 4: The Code for Your Chest X-Ray Task

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Data Preparation ---
base_dir = './chest_xray'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# --- 2. Building the CNN Model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.summary()

# --- 3. Compiling the Model ---
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- 4. Training the Model ---
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_chest_xray_model.h5', monitor='val_loss', save_best_only=True)

EPOCHS = 20

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint]
)

# --- 5. Evaluation and Reporting ---
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

y_pred_prob = model.predict(test_generator, steps=test_generator.samples)
y_pred = (y_pred_prob > 0.5).astype("int32").reshape(-1)
y_true = test_generator.classes
class_names = list(train_generator.class_indices.keys())

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
```

---

You now have both the conceptual foundation and the working code for using CNNs on real medical image data. Once comfortable, you can try deeper networks, data augmentation, or even pretrained models like ResNet.
