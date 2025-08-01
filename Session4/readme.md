Your Guide to Advanced Neural Network Training

Welcome to the next step in your computer vision journey! This guide will demystify the concepts in your second session. We'll explore the engine of neural network learning—optimizers—and cover powerful techniques like callbacks and data generators that are essential for any real-world project.

---

### Part 1: The Core Concepts (The "How" of Learning)

The magic of a neural network is its ability to adjust its own parameters to get better at a task. This process is called training or optimization. Your new PDF, "Backpropagation and optimizers," explains this beautifully.

#### 1.1. Backpropagation and Gradient Descent

**Forward Pass:** Data flows through the network, and the model makes a prediction.

**Calculate Error (Loss):** We compare the model's prediction to the true label using a loss function (like Cross-Entropy). This gives us a single number that tells us how wrong the model was.

**Backward Pass (Backpropagation):** The error is propagated backward through the network using calculus (specifically, the chain rule). We calculate the gradient for each weight and bias.

**Update Weights:** We move in the opposite direction of the gradient:

```python
w_new = w_old - eta * ∇L
```

* `w`: the weight
* `eta`: learning rate
* `∇L`: the gradient of the loss

This is **Gradient Descent**: like taking small steps downhill on a foggy mountain to reach the lowest valley (minimum loss).

#### 1.2. The Three Flavors of Gradient Descent

* **Batch Gradient Descent:**

  * Uses entire dataset
  * Smooth but slow

* **Stochastic Gradient Descent (SGD):**

  * Uses one random data point
  * Fast, noisy

* **Mini-Batch Gradient Descent:**

  * Uses small batches (e.g., 32)
  * Most practical

#### 1.3. Smarter Optimizers: The Adaptive Methods

* **SGD with Momentum**: Adds inertia to updates
* **Adagrad**: Adjusts learning rates per parameter
* **RMSProp**: Solves Adagrad's decay issue
* **Adam**: Combines momentum + RMSProp

Adam is the standard default optimizer today.

---

### Part 2: The Code, Section by Section

#### 2.1. Setup and One-Hot Encoding

```python
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

#### 2.2. Comparing Gradient Descent Methods

```python
model.fit(..., batch_size=len(X_train), ...)   # Batch
model.fit(..., batch_size=1, ...)              # SGD
model.fit(..., batch_size=32, ...)             # Mini-Batch
```

#### 2.3. Exploring Advanced Optimizers

```python
# Momentum
optimizer=tf.optimizers.SGD(learning_rate=0.01, momentum=0.9)
# Adagrad
optimizer=tf.optimizers.Adagrad(learning_rate=0.01)
# RMSProp
optimizer=tf.optimizers.RMSprop(learning_rate=0.01)
# Adam
optimizer=tf.optimizers.Adam(learning_rate=0.01)
```

#### 2.4. Deeper Evaluation: Beyond Accuracy

```python
from sklearn.metrics import confusion_matrix, classification_report

# Predictions
y_pred = model.predict(X_test).argmax(axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')

# Report
print(classification_report(y_true, y_pred))
```

#### 2.5. Making Training Smarter: Callbacks

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

model.fit(..., callbacks=[early_stopping, model_checkpoint, reduce_lr])
```

#### 2.6. Handling Imbalanced Data: class\_weight

```python
from sklearn.utils.class_weight import compute_class_weight
model.fit(..., class_weight=class_weights_dic)
```

#### 2.7. Loading Image Data from Directories

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(..., subset='training')
val_generator = train_datagen.flow_from_directory(..., subset='validation')

model.fit(train_generator, validation_data=val_generator, ...)
```

#### 2.8. Saving and Loading Your Work

```python
# Save
model.save('my_model.h5')
# Load
model = load_model('my_model.h5')

# Save weights only
model.save_weights('my_model_weights.weights.h5')
# Load weights
model.load_weights('my_model_weights.weights.h5')
```

---

### Part 3: How to Apply This Yourself

* Use **Adam** as your default optimizer
* Always use **mini-batch** with batch\_size=32
* Use **callbacks** like EarlyStopping and ModelCheckpoint
* Evaluate with **confusion matrix** and **classification report**
* Use **ImageDataGenerator** for large datasets
* Always **experiment** and visualize results

You've now covered the tools and techniques that are used every day by professionals. Keep practicing with them, and you'll build a strong, practical foundation in deep learning.
