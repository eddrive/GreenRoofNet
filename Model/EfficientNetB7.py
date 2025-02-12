import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB7
from sklearn.model_selection import train_test_split

# Load compressed dataset stored in NPZ format
npzfile = np.load('dataset/preprocessed/dataset_compressed.npz')
X = npzfile['images']  # Feature images
Y = npzfile['masks']  # Corresponding segmentation masks

# Split dataset into train (70%), validation (15%), and test (15%)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Print dataset shapes for verification
print(f"Training set: X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"Validation set: X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
print(f"Test set: X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

# Function to build a segmentation model using EfficientNetB7 as the backbone
def build_model(input_shape=(600, 600, 3)):
    # Load EfficientNetB7 with pretrained weights (ImageNet), excluding top layers
    base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the backbone to retain pretrained features

    # Define the model architecture
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # Use the base model in inference mode
    
    # Add transpose convolutional layers to upsample the feature maps
    x = layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu')(x)

    # Crop the output to match the target size
    x = layers.Cropping2D(((4, 4), (4, 4)))(x)

    # Output layer with sigmoid activation for binary segmentation
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    # Define and return the model
    model = keras.Model(inputs, outputs)
    return model

# Compile the model
model = build_model()

# Define training parameters
optimizer = 'adam'  # Adam optimizer
loss = 'binary_focal_loss'  # Binary focal loss to handle class imbalance
metrics = ['accuracy']  # Accuracy metric
batch_size = 20  # Number of samples per batch
epochs = 100  # Maximum number of training epochs

# Compile the model with Binary Focal Crossentropy loss
model.compile(optimizer=optimizer, loss=keras.losses.BinaryFocalCrossentropy(
            gamma=2.0, from_logits=True), metrics=metrics)

# Define early stopping callback to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=[early_stopping]
)

# Save the trained model with a descriptive filename
model.save(f'model_effb7_{optimizer}_{loss}_{X_train.shape[0]}_{batch_size}_{epochs}.keras')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Test Accuracy: {test_acc:.4f}')

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

# Display sample predictions from the test set
plot_predictions(model, X_test, Y_test)
