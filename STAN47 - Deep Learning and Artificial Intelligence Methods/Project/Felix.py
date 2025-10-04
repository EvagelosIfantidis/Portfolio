num_classes = 7

model_base = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(180, 180, 3)),
  tf.keras.layers.Rescaling(1./255),

  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),

  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(32,activation='relu'),
  tf.keras.layers.Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
])


model_base.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history_base=model_base.fit(
  train_ds,
  validation_data=val_ds,
  batch_size=32,
  epochs=6
)

num_classes = 7

model_dropout = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(180, 180, 3)),
  tf.keras.layers.Rescaling(1./255),

  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.5),

  tf.keras.layers.Conv2D(32, 3, activation='relu'),

  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.5),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dropout(0.5),

  tf.keras.layers.Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
])


model_dropout.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history_dropout=model_dropout.fit(
  train_ds,
  validation_data=val_ds,
  batch_size=32,
  epochs=6
)


import matplotlib.pyplot as plt

# Extract accuracy values from both models
train_acc_base = history_base.history['accuracy']
val_acc_base = history_base.history['val_accuracy']

train_acc_dropout = history_dropout.history['accuracy']
val_acc_dropout = history_dropout.history['val_accuracy']

epochs_range = range(1, len(train_acc_dropout) + 1)

# Plot accuracy for both models
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_acc_dropout, label='Regularized Model - Training Accuracy', marker='o', linestyle='-')
plt.plot(epochs_range, val_acc_dropout, label='Regularized Model - Validation Accuracy', marker='o', linestyle='--')

plt.plot(epochs_range, train_acc_base, label='Unregularized Model - Training Accuracy', marker='s', linestyle='-')
plt.plot(epochs_range, val_acc_base, label='Unregularized Model - Validation Accuracy', marker='s', linestyle='--')

# Labels and title
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy (Regularized vs Unregularized Model)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()