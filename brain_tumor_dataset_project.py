import tensorflow as tf
import matplotlib.pyplot as plt
import os
# Configure plotting
plt.rc('image', cmap='gray')
plt.rc('figure', autolayout=True)

# Step 2: Load and Preprocess the Image
Train_dir = "Training"
sample_path = os.path.join(Train_dir, os.listdir(Train_dir)[1])
image_path =  os.path.join(sample_path, os.listdir(sample_path)[0])

# Read image
image = tf.io.read_file(image_path)

# Convert image to grayscale
image = tf.io.decode_jpeg(image, channels=1)

# Resize image
image = tf.image.resize(image, [300, 300])

# Normalize pixel values
image = tf.image.convert_image_dtype(image, tf.float32)

print("Original Image Shape:", image.shape)

# Display image
plt.figure(figsize=(5, 5))
plt.imshow(tf.squeeze(image))
plt.title("Original Image")
plt.axis('off')
plt.show()

# Add batch dimension
image = tf.expand_dims(image, axis=0)

# Step 3: Define Convolution Kernel
# A edge detection filter
kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=tf.float32)

kernel = tf.reshape(kernel, [3, 3, 1, 1])

# Step 4: Apply Convolution Layer
conv_output = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=[1, 1, 1, 1],
    padding='SAME'
)

print("After Convolution Shape:", conv_output.shape)

plt.figure(figsize=(5, 5))
plt.imshow(tf.squeeze(conv_output))
plt.title("After Convolution")
plt.axis('off')
plt.show()

# Step 5: Apply ReLU Activation Function 
relu_output = tf.nn.relu(conv_output)

print("After ReLU Shape:", relu_output.shape)

plt.figure(figsize=(5, 5))
plt.imshow(tf.squeeze(relu_output))
plt.title("After ReLU Activation")
plt.axis('off')
plt.show()

# Step 6: Apply Max Pooling Layer 
pool_output = tf.nn.max_pool2d(
    input=relu_output,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME'
)

print("After Pooling Shape:", pool_output.shape)

plt.figure(figsize=(5, 5))
plt.imshow(tf.squeeze(pool_output))
plt.title("After Max Pooling")
plt.axis('off')
plt.show()

# Step 7: Apply Flatten Layer
flatten_layer = tf.keras.layers.Flatten()
flatten_output = flatten_layer(pool_output)

print("After Flatten Shape:", flatten_output.shape)
print("First 20 Flattened Values:")
print(flatten_output.numpy()[0][:20])

# Step 8: Add Fully Connected (Dense) Layer 
dense_layer = tf.keras.layers.Dense(
    units=64,
    activation='relu'
)

dense_output = dense_layer(flatten_output)

print("After Fully Connected Layer Shape:", dense_output.shape)
print("Dense Layer Output:")
print(dense_output.numpy())

# Performance Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Example Labels for demonstration 
y_true = [1, 0, 1, 1, 0, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 1]

# Calculation
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
