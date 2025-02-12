import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split

# Load compressed dataset stored in NPZ format
npzfile = np.load('Data.npz')
X = npzfile['images']  # Feature images
Y = npzfile['masks']  # Corresponding segmentation masks

# Split dataset into train (70%), validation (15%), and test (15%)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Print dataset shapes for verification
print(f"Training set: X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"Validation set: X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
print(f"Test set: X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

# Function to create DeepLabV3+ segmentation model
def deeplabv3_plus(input_shape=(600, 600, 3), num_classes=1):
    # Load ResNet50 with pretrained weights (ImageNet), excluding top layers
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the backbone to retain pretrained features

    # Extract features from the last convolutional block
    x = base_model.get_layer('conv4_block6_out').output

    # ASPP (Atrous Spatial Pyramid Pooling) block
    aspp = layers.Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    aspp = layers.BatchNormalization()(aspp)
    aspp = layers.ReLU()(aspp)

    aspp = layers.Conv2D(256, (3, 3), dilation_rate=6, padding="same", use_bias=False)(aspp)
    aspp = layers.BatchNormalization()(aspp)
    aspp = layers.ReLU()(aspp)

    aspp = layers.Conv2D(256, (3, 3), dilation_rate=12, padding="same", use_bias=False)(aspp)
    aspp = layers.BatchNormalization()(aspp)
    aspp = layers.ReLU()(aspp)

    aspp = layers.Conv2D(256, (3, 3), dilation_rate=18, padding="same", use_bias=False)(aspp)
    aspp = layers.BatchNormalization()(aspp)
    aspp = layers.ReLU()(aspp)

    # Upsampling layers to reconstruct the segmentation mask
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(aspp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Crop the output to match the target size
    x = layers.Cropping2D(((4, 4), (4, 4)))(x)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)
    model = keras.Model(inputs=base_model.input, outputs=outputs)
    return model

# Create and compile the model
deep_model = deeplabv3_plus()
deep_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

deep_model.summary()

# Normalize images in-place
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
X_test = preprocess_input(X_test)

# Early stopping callback to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = deep_model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=8,
    epochs=50,
    verbose=1,
    callbacks=[early_stopping]
)

# save the model with a descriptive filename
deep_model.save(f'model_deeplabv3plus_{X_train.shape[0]}_{50}.keras')

# Function to visualize model predictions
def plot_predictions(model, X, Y, num_samples=3):
    indices = random.sample(range(len(X)), num_samples)  # Randomly select sample indices
    preds = model.predict(X[indices])  # Get model predictions
    preds = (preds > 0.5).astype(np.uint8)  # Convert probabilities to binary masks

    # Plot original images, true masks, and predicted masks
    plt.figure(figsize=(10, num_samples * 3))
    for i, idx in enumerate(indices):
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(X[idx])
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(Y[idx].squeeze(), cmap='gray')
        plt.title('True Mask')
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(preds[i].squeeze(), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
    plt.show()

# Show sample predictions from the test set
plot_predictions(deep_model, X_test, Y_test)
