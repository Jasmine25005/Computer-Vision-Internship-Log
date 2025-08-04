# **Computer Vision Fundamentals: A Detailed Guide**

Welcome to your computer vision journey! This document will serve as your comprehensive guide, bridging the gap between the theoretical concepts and the practical hands-on code.

---

## **1. The Big Picture: What is Computer Vision?**

Before diving into the code. The Computer Vision is a part of Deep Learning, which is a part of Machine Learning, all under the umbrella of Artificial Intelligence.

* **Artificial Intelligence (AI):**
  The broad concept of creating machines that can think or act intelligently, like humans. Alan Turing's test is a famous thought experiment to determine if a machine is "intelligent."

* **Machine Learning (ML):**
  A subset of AI where we don't program explicit rules. Instead, we "train" a model by showing it lots of data, and it learns the patterns itself.

* **Deep Learning (DL):**
  A specialized type of ML that uses complex, multi-layered "neural networks." It's been the driving force behind recent breakthroughs like self-driving cars (Waymo), game-playing AI (AlphaGo), and image generation (DALL-E).

* **Computer Vision (CV):**
  This is our focus. It's a field of AI that trains computers to "see" and understand the visual world, it's different from Image Processing.

### Image Processing vs. Computer Vision

* **Image Processing:**
  The input is an image, and the output is a modified image (e.g., making it black and white, increasing brightness, applying a filter).

* **Computer Vision:**
  The input is an image, and the output is information or understanding about the image (e.g., "This is a cat," "There are 3 cars in this scene," "This is a cancerous cell.").

---

## **2. The Tools of the Trade: Python Libraries for CV**

In your notebook, you use four key libraries. Each has its strengths, and they are often used together.

---

### **How is an Image Represented Digitally?**

An image is just a grid of tiny dots called **pixels**. For a computer, this grid is represented as a multi-dimensional array of numbers.

* **Dimensions:**
  The height and width of the grid (e.g., 800x600 pixels).

* **Channels:**
  Each pixel has a value representing its color. For color images, we typically use three channels: **Red**, **Green**, and **Blue (RGB)**.

* **Pixel Value:**
  Typically ranges from 0 (black) to 255 (full color intensity).

* **Color image:** 3D array → `(height, width, 3)`

* **Grayscale image:** 2D array → `(height, width)`

---

### **Library 1: Pillow (PIL)**

**What it is:**
The Python Imaging Library (PIL), and its modern fork **Pillow**, is great for basic image manipulation tasks.

**When to use it:**
Excellent for simple, high-level image file operations.

**Code Breakdown:**

```python
# !pip install pillow
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = Image.open('/content/image_3.jpg')
plt.imshow(img)

img1 = np.array(img)
print(type(img1))  # -> <class 'numpy.ndarray'>
print(img1.shape)
```

**Key Takeaway:**
Pillow opens images into its own object. Convert it using `np.array()` for use in other libraries.

---

### **Library 2: Matplotlib**

**What it is:**
Primarily used for plotting, but can also read and display images.

**When to use it:**
Great for visualization.

**Code Breakdown:**

```python
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img = mpimg.imread('/content/image_3.jpg')
print(type(img))
print(img.shape)
plt.imshow(img)
plt.colorbar()
```

**Comparison:**
Pillow → Manual conversion to NumPy.
Matplotlib → Already in NumPy format.

---

### **Library 3: Scikit-image**

**What it is:**
A powerful library for image processing algorithms.

**When to use it:**
For filtering, segmentation, normalization, etc.

**Code Breakdown:**

```python
# !pip install scikit-image
from skimage import io, img_as_float
import matplotlib.pyplot as plt

image = io.imread('/content/image_3.jpg')
print(type(image))
plt.imshow(image)

image_float = img_as_float(image)
# OR
# image_float = image.astype(np.float64)/255
```

**Why Normalize?**
ML models perform better with values in `[0.0, 1.0]`.

---

### **Library 4: OpenCV**

**What it is:**
The industry standard for Computer Vision. Fast, powerful, and widely adopted.

**When to use it:**
Almost any serious CV task.

**Code Breakdown:**

```python
# !pip install opencv-python
import cv2

img = cv2.imread('/content/image_3.jpg')
print(type(img))
plt.imshow(img)  # Will look wrong!

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)

img_resized = cv2.resize(img, (224, 224))
print(img_resized.shape)

img_gray = cv2.imread('/content/image_3.jpg', 0)
print(img_gray.shape)
plt.imshow(img_gray, cmap='gray')

from skimage import filters
edge_sobel = filters.sobel(img_gray)
plt.imshow(edge_sobel, cmap='gray')
```

**Note:**
OpenCV uses BGR by default → convert to RGB for display.

---

## **3. Working with Image Data**

### **Manipulating Pixels**

```python
random_image = np.random.random([500, 500])
plt.imshow(random_image, cmap='gray')

img_rgb[10:75, 10:75] = (0, 0, 0)
plt.imshow(img_rgb)
```

---

### **Reading Multiple Images (Building a Dataset)**

#### Method 1: `os.path.join()`

```python
import os
os.path.join('data', 'cats', 'cat.jpg')
```

#### Method 2: `glob.glob()`

```python
import glob
glob.glob('/content/dataset/cats/*.jpg')
image_paths = glob.glob('/content/dataset/*/*.jpg')
```

#### Method 3: `os.walk()`

```python
for root, dirs, files in os.walk('/content/dataset'):
    print(f"Current Folder: {root}")
    print(f"Sub-folders in it: {dirs}")
    print(f"Files in it: {files}")
    print("---")
```

---

### **Practical Dataset Loading Code**

#### Example A: Using `os.listdir`

```python
images = []
labels = []
root_dir = '/content/dataset'

for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        for image_file in os.listdir(folder_path):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path)
                images.append(image)
                labels.append(folder_name)
```

#### Example B: Using `glob`

```python
import glob

image_paths = glob.glob('/content/dataset/*/*.jpg')
images = [cv2.imread(path) for path in image_paths]
labels = [path.split('/')[-2] for path in image_paths]
```

**Pro Tip:**
Use `glob + list comprehension` for clean, Pythonic code.

---

## **4. Final Data Preparation: Encodings**

### `enumerate` and `zip`

```python
animals = ['cat', 'dog', 'horse']
for idx, name in enumerate(animals):
    print(idx, name)

label_to_index = {name: idx for idx, name in enumerate(animals)}
print(label_to_index)
```

---

## **5. Your Homework Task: Putting It All Together**

```python
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

image_paths = glob.glob('/content/dataset/*/*.jpg')
class_names = sorted(list(set([path.split('/')[-2] for path in image_paths])))
label_to_index = {name: idx for idx, name in enumerate(class_names)}

images = []
labels = []

for path in image_paths:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (128, 128))
    image_normalized = image_resized.astype(np.float32) / 255.0
    label_name = path.split('/')[-2]
    label_index = label_to_index[label_name]
    images.append(image_normalized)
    labels.append(label_index)

images = np.array(images)
labels = np.array(labels)

print(f"Shape of images array: {images.shape}")
print(f"Shape of labels array: {labels.shape}")

plt.imshow(images[4])
plt.title(f"Label: {labels[4]}")
plt.show()
```

---
---
## The Modern Way: Loading Images with tf.keras.utils.image_dataset_from_directory
Earlier, in this Session, you learned how to load images manually using libraries like glob and os to find file paths and then looping through them to read each image with OpenCV. While that method is fundamental to understanding the process, it's not very efficient for large datasets.

TensorFlow and Keras provide a powerful utility that does all the heavy lifting for you in one line of code.

What is image_dataset_from_directory?
It's a function that reads a directory of images, which is sorted into class-specific subdirectories, and creates a tf.data.Dataset object. This object is highly optimized for performance and is the standard way to feed data into a Keras model for training.

Why is it better than the manual method?

Automation: It automatically finds the images, resizes them, creates labels from the folder names, and shuffles the data.

Memory Efficiency: It doesn't load all the images into memory at once. Instead, it loads them in batches from the disk as needed, which is essential for working with huge datasets that don't fit in your RAM.

Performance: The tf.data.Dataset object it creates has powerful methods like .cache() and .prefetch() that can dramatically speed up your model training pipeline.

Simplicity: It replaces 15-20 lines of manual code with a single function call.

How It Works: The Directory Structure
The most important requirement for this function is that your images must be organized in a specific way. You need a main directory, and inside it, one subdirectory for each class.

For example:

## /content/dataset/
## ├── cats/
## │   ├── cat_1.jpg
## │   ├── cat_2.jpg
## │   └── ...
## ├── dogs/
## │   ├── dog_1.jpg
## │   ├── dog_2.jpg
## │   └── ...
## └── horses/
##     ├── horse_1.jpg
##     ├── horse_2.jpg
##     └── ...
The function will automatically:

Identify cats, dogs, and horses as the class names.

Assign an integer label to each class (e.g., cats: 0, dogs: 1, horses: 2).

Pair each image with its correct label.

Practical Example
Let's see how to use it to load the dataset structure shown above and prepare it for training.

Python

import tensorflow as tf
import matplotlib.pyplot as plt

# Define main parameters
DATASET_PATH = '/content/dataset'
IMAGE_SIZE = (128, 128) # The size to resize all images to
BATCH_SIZE = 32 # How many images to load in each batch

# --- Create the Training Dataset ---
# This will load 80% of the images for training.
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels='inferred',           # Automatically infer labels from folder names
    label_mode='int',            # Labels will be integers (0, 1, 2...)
    image_size=IMAGE_SIZE,       # Resize all images to 128x128
    batch_size=BATCH_SIZE,       # Group images into batches of 32
    validation_split=0.2,        # Reserve 20% of the data for validation
    subset='training',           # Specify that this is the training subset
    seed=42                      # Seed for shuffling and splitting ensures consistency
)

# --- Create the Validation Dataset ---
# This will load the remaining 20% of the images for validation.
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels='inferred',
    label_mode='int',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,        # Must be the same split as training
    subset='validation',         # Specify that this is the validation subset
    seed=42                      # Must be the same seed as training
)

# --- Inspect the Dataset ---
# You can easily see the class names it found
class_names = train_dataset.class_names
print("Class names found:", class_names)
# Expected output: Class names found: ['cats', 'dogs', 'horses']

# Let's look at one batch from the training dataset
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):  # Take just the first batch
    print("Shape of images batch:", images.shape) # (Batch Size, Height, Width, Channels)
    print("Shape of labels batch:", labels.shape) # (Batch Size,)

    # The images are loaded as TensorFlow Tensors, not NumPy arrays.
    # The pixel values are already floats from 0 to 255.
    # We can display them.
    for i in range(9): # Display the first 9 images of the batch
        ax = plt.subplot(3, 3, i + 1)
        # We need to convert the tensor to a NumPy array and ensure it's an integer for display
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()

# --- Optional: Normalize the data ---
# A common next step is to normalize the pixel values from [0, 255] to [0, 1]
# We can do this very efficiently using the .map() method.
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_train_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))

Key Takeaway
The image_dataset_from_directory function is a high-level utility that bridges the gap between your organized image folders on disk and a high-performance tf.data.Dataset ready for model training. It handles labeling, resizing, batching, and splitting automatically, making it the preferred method for any TensorFlow/Keras image classification project.
