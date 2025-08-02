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

Go through this explanation carefully and make sure you understand how each part contributes to the final goal of preparing your data for a machine learning model.

---

