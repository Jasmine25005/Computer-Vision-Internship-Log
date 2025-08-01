---

### **Session 2 Guide: Feature Extraction, Matching, and Filtering (Beginner's Guide)**

Welcome to the detailed explanation for Session 2. This guide is designed for beginners and focuses on the advanced concepts of how a computer understands an image by identifying its important "features." We will start from the very basics, including a deep dive into image filtering, which is a critical first step.

---

### **Part 1: Image Filtering & Noise Reduction**

Before we can analyze an image, it's often full of imperfections or "noise" (like the graininess in old photos). To "clean" the image, we apply filters that smooth it out. The core technique behind this is 2D Convolution.

---

### **What is a Kernel? (The Heart of Filtering)**

Think of a Kernel as a tiny magnifying glass with a special lens that you slide over your image. It's a small matrix (e.g., 3x3) of numbers. The numbers on this "lens" determine what effect it will have on the image.

* A kernel designed for blurring will have numbers that average out the pixels.
* A kernel designed for sharpening will have numbers that exaggerate the differences between pixels.
* A kernel for edge detection will have numbers that highlight where colors change abruptly.

The kernel is the "recipe" for the filtering effect you want to achieve.

---

### **What is 2D Convolution? (The Process of Filtering)**

Convolution is the process of applying the kernel to the image.

1. You place the kernel (3x3 grid) on top of a 3x3 section of your image.
2. Multiply the kernel's numbers with the image's pixel values underneath.
3. Sum all 9 results.
4. That sum becomes the new center pixel.
5. Slide the kernel and repeat across the entire image.

This sliding action "convolves" the kernel across the image.

---

### **Manual 2D Convolution**

```python
kernel = np.ones((5, 5), np.float32) / 25
img_filtered = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
```

---

### **Common Blurring Filters**

**Averaging Blur**

```python
cv2.blur(src=image, ksize=(5, 5))
```

**Gaussian Blur**

```python
cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX=0)
```

**Median Blur**

```python
cv2.medianBlur(src=image, ksize=5)
```

**Bilateral Filtering**

```python
cv2.bilateralFilter(src=image, d=9, sigmaColor=75, sigmaSpace=75)
```

---

### **Part 2: Edge Detection**

Edges are boundaries of objects. Edge detectors find places where pixel intensity changes sharply.

---

### **What is a "Derivative" in an Image?**

* First derivative: Difference between adjacent pixels.
* Second derivative: Change in the rate of change — high at edge centers.

---

### **1. Laplacian Operator**

```python
blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)
```

---

### **2. Sobel Operator**

```python
Gx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
G = cv2.bitwise_or(np.uint8(np.absolute(Gx)), np.uint8(np.absolute(Gy)))
```

---

### **3. Canny Edge Detector**

```python
canny = cv2.Canny(gray_image, 100, 200)
```

---

### **Part 3: Feature Detection & Description**

---

### **What is a "Feature"?**

A distinctive point like a corner, edge, or blob.

---

### **1. Corner Detectors**

**Harris Corner Detector**

```python
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)
img[dst > 0.01 * dst.max()] = [0, 0, 255]
```

**FAST**

```python
fast = cv2.FastFeatureDetector_create()
kp = fast.detect(img, None)
img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
```

---

### **2. Blob Detectors (LoG / DoG)**

```python
from skimage.feature import blob_dog, blob_log

blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=0.1)
blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=0.1)
```

---

### **3. SIFT (Scale-Invariant Feature Transform)**

```python
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(img, None)
print(des.shape)
img_with_keypoints = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```

---

### **Part 4: Feature Matching**

---

### **1. Brute-Force Matcher**

```python
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
```

---

### **2. Lowe's Ratio Test**

```python
good_matches = []
for m, n in matches:
  if m.distance < 0.75 * n.distance:
    good_matches.append(m)

result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```

---

### **Part 5: Practical Application - Handwritten Digit Recognition**

---

### **1. HOG (Histogram of Oriented Gradients)**

```python
hog_feat = hog(image,
               orientations=9,
               pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               block_norm='L2-Hys')
```

---

### **2. Classification Pipeline (HOG + SVM)**

* Load MNIST dataset
* Extract HOG features
* Train an SVM classifier
* Evaluate accuracy

---


