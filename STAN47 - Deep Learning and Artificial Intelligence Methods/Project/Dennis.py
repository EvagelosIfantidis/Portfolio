

num_classes = 7

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(180, 180, 3)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


# %%
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['accuracy'])



# %%


# Generate a filename based on model architecture
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
architecture_summary = f"CNN_32-64-128_Dense-128"

log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"training_{architecture_summary}_{timestamp}.log")

# Create a subdirectory for saving confusion matrices
conf_matrix_dir = os.path.join(log_dir, f"confusion_matrices_{timestamp}")
os.makedirs(conf_matrix_dir, exist_ok=True)

# Logging function
def log_to_file(message):
    with open(log_filename, "a") as f:
        f.write(message + "\n")
    print(message)  # Also print to console

# Function to log epoch results and save confusion matrices
def log_epoch_results(epoch, logs):
    log_to_file(f"Epoch {epoch+1}: Loss={logs['loss']:.4f}, Accuracy={logs['accuracy']:.4f}, "
                f"Val_Loss={logs['val_loss']:.4f}, Val_Accuracy={logs['val_accuracy']:.4f}")
    
    # Generate predictions for validation set
    y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)  # Assuming val_ds is a TensorFlow dataset
    y_pred = np.argmax(model.predict(val_ds), axis=-1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot and save confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(len(cm)), yticklabels=np.arange(len(cm)))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Epoch {epoch+1}")

    cm_filename = os.path.join(conf_matrix_dir, f"confusion_matrix_epoch_{epoch+1}.png")
    plt.savefig(cm_filename)
    plt.close()
    
# Save model architecture to log
with open(log_filename, "w") as f:
    f.write("Model Architecture:\n")
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# Training loop with interruption handling
try:
    log_to_file("\nTraining started...\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: log_epoch_results(epoch, logs)
        )]
    )
    log_to_file("\nTraining completed successfully.")

except KeyboardInterrupt:
    log_to_file("\nTraining interrupted by user. Partial results saved.")




# %%
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_layer(image, layer_index, model, num_col=8):
#     """
#     Visualizes the feature maps of a specific layer in the CNN.

#     Parameters:
#     - image : The input image to process (from train_ds).
#     - layer_index: The index of the layer in the model.
#     - model: The trained CNN model.
#     - num_col: The number of subplots per row.
#     """
#     # Better check for model initialization
#     if not model.built:
#         raise ValueError("Model has not been built. Please ensure the model has been compiled and trained.")

#     # Extract feature maps
#     conv_pool_layers = [layer for layer in model.layers if 'conv' in layer.name or 'pool' in layer.name]
#     if layer_index >= len(conv_pool_layers):
#         raise ValueError(f"Layer index {layer_index} is out of range. Model has {len(conv_pool_layers)} conv/pool layers.")
    
#     extractor = tf.keras.Model(inputs=model.inputs,
#                               outputs=[layer.output for layer in conv_pool_layers])
#     features = extractor(tf.convert_to_tensor([image]))  # Use tf.convert_to_tensor for better compatibility

#     # Select the feature map of the chosen layer
#     layer_features = features[layer_index].numpy()[0]  # Extract feature maps from batch
#     num_features = layer_features.shape[-1]  # Number of filters
#     num_row = int(np.ceil(num_features / num_col))  # Calculate rows dynamically

#     # Plot feature maps
#     fig, axes = plt.subplots(num_row, num_col, figsize=(num_col*2, num_row*2))
#     axes = axes.flatten()  # Flatten in case of 1 row
#     for i in range(num_features):
#         axes[i].imshow(layer_features[..., i], cmap='gray')
#         axes[i].axis('off')
#     for j in range(i+1, len(axes)):  # Hide unused subplots
#         axes[j].axis('off')
    
#     plt.suptitle(f'Feature Maps of Layer {layer_index} ({model.layers[layer_index].name})', fontsize=14)
#     plt.show()

# # ✅ Select an image from the training dataset
# image_batch, labels_batch = next(iter(train_ds))  # Get a batch of images
# sample_image = image_batch[0]  # Select the first image

# # ✅ Visualize feature maps from the first few layers
# print("🔹 Feature maps of the first convolutional layer:")
# plot_layer(sample_image, layer_index=0, model=model)

# print("🔹 Feature maps of the second convolutional layer:")
# plot_layer(sample_image, layer_index=2, model=model)

# print("🔹 Feature maps of the third convolutional layer:")
# plot_layer(sample_image, layer_index=4, model=model)


# %% [markdown]
# Drop-out

# %%
