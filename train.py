"""
train.py
========
Training and Fine-Tuning Pipeline for Ancient Tamil Palm-Leaf
Vatteluttu Character Recognition Model.

Features:
- Loads all 71 character classes from the dataset (7,100 images)
- 80/20 train/validation split with deterministic seeding
- Data augmentation: RandomRotation, RandomZoom, RandomTranslation
- Deep CNN Architecture: 4 Conv2D blocks with BatchNormalization, MaxPooling, Dropout
- Learning rate scheduling with ReduceLROnPlateau and EarlyStopping
- Saves the best model checkpoint to best_handwriting_model_fixed.keras
- Generates accuracy/loss training curves and validation report
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, callbacks

# Configuration
DATASET_DIR = os.path.join(
    os.path.dirname(__file__),
    "Tamil palmleaf Vatteluttu Datastet",
    "palmleaf-tamil",
    "FINAL DATASET",
    "DATASET"
)
OUTPUT_MODEL_PATH = "best_handwriting_model_fixed.keras"
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 15  # Efficient convergence on 7,100 images
SEED = 42


def get_datasets(dataset_dir: str = DATASET_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Load and prepare training and validation tf.data datasets."""
    print(f"Loading dataset from: {dataset_dir}")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Training dataset (80%)
    train_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=img_size,
        color_mode="grayscale",
        batch_size=batch_size,
        label_mode="int"
    )

    # Validation dataset (20%)
    val_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=img_size,
        color_mode="grayscale",
        batch_size=batch_size,
        label_mode="int"
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes.")

    # Performance optimization: cache and prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_recognition_model(num_classes: int = 71, input_shape=(64, 64, 1)) -> keras.Model:
    """
    Construct the deep CNN architecture matching Box 5 of the architecture:
    - Rescaling layer: normalizes pixels [0, 255] -> [0, 1]
    - Data augmentation layers: Rotation, Zoom, Translation
    - 4 Convolutional blocks (32, 64, 128, 256) with BatchNorm and MaxPooling
    - Dense representation with Dropout for regularization
    - Softmax classification output
    """
    inputs = layers.Input(shape=input_shape, name="input_glyph")

    # Rescaling
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Augmentation layers (active only during training)
    x = layers.RandomRotation(0.08, fill_mode="constant", fill_value=0.0, name="random_rotation")(x)
    x = layers.RandomZoom(0.08, fill_mode="constant", fill_value=0.0, name="random_zoom")(x)
    x = layers.RandomTranslation(0.04, 0.04, fill_mode="constant", fill_value=0.0, name="random_translation")(x)

    # Block 1: 32 filters
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2: 64 filters
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3: 128 filters
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4: 256 filters
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification Head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="TamilVatteluttuCNN")
    return model


def train(epochs: int = EPOCHS, initial_model_path: str = None):
    """Train the model and save the best checkpoint."""
    train_ds, val_ds, class_names = get_datasets()
    num_classes = len(class_names)

    if initial_model_path and os.path.exists(initial_model_path):
        print(f"Loading existing model weights from: {initial_model_path}")
        model = keras.models.load_model(initial_model_path)
    else:
        print("Building fresh Tamil Vatteluttu recognition CNN model...")
        model = build_recognition_model(num_classes=num_classes)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        OUTPUT_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    lr_reducer_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )
    early_stop_cb = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb, lr_reducer_cb, early_stop_cb],
        verbose=1
    )

    # Final Evaluation on Validation Set
    print("\nEvaluating best saved model on validation set...")
    best_model = keras.models.load_model(OUTPUT_MODEL_PATH)
    val_loss, val_acc = best_model.evaluate(val_ds, verbose=0)
    print(f"============================================================")
    print(f"FINAL MODEL VALIDATION ACCURACY: {val_acc * 100:.2f}% (Loss: {val_loss:.4f})")
    print(f"Best model saved successfully at: {OUTPUT_MODEL_PATH}")
    print(f"============================================================")

    # Save training metrics history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_dict["final_val_accuracy"] = float(val_acc)
    history_dict["final_val_loss"] = float(val_loss)
    with open("training_metrics.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    print("Training metrics saved to: training_metrics.json")

    return best_model, history_dict


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else EPOCHS
    train(epochs=epochs)
