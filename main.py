import os
import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def prepare(test_size=0.2, random_state=42):
    df = pd.read_csv("data/data.csv", sep=';')

    samples = []
    for _, row in df.iterrows():
        samples.append({
            "id": int(row["ID"]),
            "leftImage": f"data/train/{row['ID']}_left.jpg",
            "rightImage": f"data/train/{row['ID']}_right.jpg",
            "diagnosis": [
                int(row["N"]),
                int(row["D"]),
                int(row["G"]),
                int(row["C"]),
                int(row["A"]),
                int(row["H"]),
                int(row["M"]),
                int(row["O"]),
            ]
        })

    train_samples, test_samples = train_test_split(
        samples,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    os.makedirs("data", exist_ok=True)

    with open("data/train_dataset.json", "w") as f:
        json.dump(train_samples, f, indent=4)

    with open("data/test_dataset.json", "w") as f:
        json.dump(test_samples, f, indent=4)

    return len(train_samples), len(test_samples)


def augment_image(image):
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_crop(
        image,
        size=[int(image.shape[0] * 0.9), int(image.shape[1] * 0.9), 3]
    )
    image = tf.image.resize(image, [224, 224])
    return image


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

    diagnosis = tf.cast(sample["diagnosis"], tf.float32)


    return {"left_input": left, "right_input": right}, diagnosis


def create_tf_dataset(
    json_path,
    batch_size=32,
    shuffle=True,
    augment=False,
    repeat=True,
    img_size=(224, 224)
):
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
            "diagnosis": tf.TensorSpec(shape=(8,), dtype=tf.int32),
        }
    )

    dataset = dataset.map(
        lambda x: parse_sample(x, img_size, augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(samples))

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, len(samples)


def build_model(img_size=(224, 224), num_labels=8):
    left_input = layers.Input(shape=(*img_size, 3), name="left_input")
    right_input = layers.Input(shape=(*img_size, 3), name="right_input")

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*img_size, 3)
    )
    base_model.trainable = True

    left_features = base_model(left_input)
    right_features = base_model(right_input)

    left_features = layers.GlobalAveragePooling2D()(left_features)
    right_features = layers.GlobalAveragePooling2D()(right_features)

    combined = layers.Concatenate()([left_features, right_features])
    combined = layers.Dense(256, activation="relu")(combined)
    combined = layers.Dropout(0.5)(combined)

    outputs = layers.Dense(num_labels, activation="sigmoid")(combined)

    model = models.Model(
        inputs={"left_input": left_input, "right_input": right_input},
        outputs=outputs
    )
    return model


if __name__ == "__main__":
    # Prepare data
    train_count, test_count = prepare()

    # Datasets
    train_dataset, train_len = create_tf_dataset(
        "data/train_dataset.json",
        batch_size=32,
        shuffle=True,
        augment=True,
        repeat=True
    )

    test_dataset, test_len = create_tf_dataset(
        "data/test_dataset.json",
        batch_size=32,
        shuffle=False,
        augment=False,
        repeat=False
    )

    steps_per_epoch = train_len // 32
    validation_steps = test_len // 32

    # Model
    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_model.keras",
            monitor="val_auc",
            mode="max",
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            restore_best_weights=True
        )
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


