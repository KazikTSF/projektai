import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# -----------------------------
# 1. Prepare dataset
# -----------------------------
def prepare(test_size=0.2, random_state=42):
    df = pd.read_csv("data/data.csv", sep=';')

    samples = []
    # Encode diagnosis as integer (0-7)
    for _, row in df.iterrows():
        # Assume exactly one of N, D, G, C, A, H, M, O is 1
        diagnosis_vector = [int(row[c]) for c in ["N", "D", "G", "C", "A", "H", "M", "O"]]
        diagnosis_int = np.argmax(diagnosis_vector)

        samples.append({
            "id": int(row["ID"]),
            "leftImage": f"data/train/{row['ID']}_left.jpg",
            "rightImage": f"data/train/{row['ID']}_right.jpg",
            "diagnosis": int(diagnosis_int)
        })

    # Stratified split by diagnosis
    diagnoses = [s["diagnosis"] for s in samples]
    train_samples, test_samples = train_test_split(
        samples,
        test_size=test_size,
        random_state=random_state,
        stratify=diagnoses
    )

    os.makedirs("data", exist_ok=True)
    with open("data/train_dataset.json", "w") as f:
        json.dump(train_samples, f, indent=4)
    with open("data/test_dataset.json", "w") as f:
        json.dump(test_samples, f, indent=4)

    return len(train_samples), len(test_samples)

# -----------------------------
# 2. Safe medical image augmentation
# -----------------------------
def augment_image(image):
    # Only horizontal flip, mild brightness/contrast, small random crop
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.resize_with_crop_or_pad(image, 240, 240)
    image = tf.image.random_crop(image, size=[224, 224, 3])
    return image

# -----------------------------
# 3. Parse one sample
# -----------------------------
def parse_sample(sample, img_size=(224, 224), augment=False):
    left = tf.io.read_file(sample["leftImage"])
    left = tf.image.decode_jpeg(left, channels=3)
    left = tf.image.resize(left, img_size)
    left = left / 255.0

    right = tf.io.read_file(sample["rightImage"])
    right = tf.image.decode_jpeg(right, channels=3)
    right = tf.image.resize(right, img_size)
    right = right / 255.0

    if augment:
        left = augment_image(left)
        right = augment_image(right)

    diagnosis = tf.cast(sample["diagnosis"], tf.int32)
    return {"left_input": left, "right_input": right}, diagnosis

# -----------------------------
# 4. Create tf.data.Dataset
# -----------------------------
def create_tf_dataset(json_path, batch_size=32, shuffle=True, augment=False, repeat=True, img_size=(224,224)):
    with open(json_path, "r") as f:
        samples = json.load(f)

    def gen():
        for sample in samples:
            yield sample

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "id": tf.TensorSpec(shape=(), dtype=tf.int32),
            "leftImage": tf.TensorSpec(shape=(), dtype=tf.string),
            "rightImage": tf.TensorSpec(shape=(), dtype=tf.string),
            "diagnosis": tf.TensorSpec(shape=(), dtype=tf.int32),
        }
    )

    dataset = dataset.map(lambda x: parse_sample(x, img_size, augment), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(samples))

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, len(samples)

# -----------------------------
# 5. Build dual-input model
# -----------------------------
def build_model(img_size=(224,224), num_classes=8):
    left_input = layers.Input(shape=(*img_size,3), name="left_input")
    right_input = layers.Input(shape=(*img_size,3), name="right_input")

    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*img_size,3))
    base_model.trainable = True  # fine-tuning allowed

    left_features = base_model(left_input)
    right_features = base_model(right_input)

    left_features = layers.GlobalAveragePooling2D()(left_features)
    right_features = layers.GlobalAveragePooling2D()(right_features)

    combined = layers.Concatenate()([left_features, right_features])
    combined = layers.Dense(256, activation="relu")(combined)
    combined = layers.Dropout(0.5)(combined)

    outputs = layers.Dense(num_classes, activation="softmax")(combined)

    model = models.Model(inputs={"left_input": left_input, "right_input": right_input}, outputs=outputs)
    return model

# -----------------------------
# 6. Training script
# -----------------------------
if __name__ == "__main__":
    # Prepare data
    train_count, test_count = prepare()

    train_dataset, train_len = create_tf_dataset("data/train_dataset.json", batch_size=32, shuffle=True, augment=True, repeat=True)
    test_dataset, test_len = create_tf_dataset("data/test_dataset.json", batch_size=32, shuffle=False, augment=False, repeat=False)

    steps_per_epoch = train_len // 32
    validation_steps = test_len // 32

    # Build and compile model
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("models/best_model.keras", monitor="val_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    ]

    # Train
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=20,
        callbacks=callbacks
    )

    # Save final model
    model.save("models/final_model.keras")
