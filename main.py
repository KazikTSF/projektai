import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# =====================
# CONFIG
# =====================
BATCH_SIZE = 32
IMG_SIZE = (400, 400)
DISEASE_NAMES = ["N", "D", "G", "C", "A", "H", "M", "O"]


# =====================
# DATA PREPARATION
# =====================
def prepare(test_size=0.2, random_state=42):
    df = pd.read_csv("/kaggle/input/ocular-disease-recognition-odir5k/full_df.csv")

    samples = []
    for _, row in df.iterrows():
        samples.append({"id": int(row["ID"]), "leftImage": f"/home/processed/{row['ID']}_left.jpg",
            "rightImage": f"/home/processed/{row['ID']}_right.jpg", "diagnosis": [int(row[c]) for c in DISEASE_NAMES]})

    train_samples, test_samples = train_test_split(samples, test_size=test_size, shuffle=True,
        random_state=random_state)

    json.dump(train_samples, open("/home/train_dataset.json", "w"), indent=2)
    json.dump(test_samples, open("/home/test_dataset.json", "w"), indent=2)

    return len(train_samples), len(test_samples)


# =====================
# AUGMENTATION
# =====================
def augment_image(image):
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.resize(image, [IMG_SIZE[0] + 20, IMG_SIZE[1] + 20])
    image = tf.image.random_crop(image, [IMG_SIZE[0], IMG_SIZE[1], 3])
    return image


def parse_sample(sample, augment):
    left = tf.image.decode_jpeg(tf.io.read_file(sample["leftImage"]), channels=3)
    right = tf.image.decode_jpeg(tf.io.read_file(sample["rightImage"]), channels=3)

    left = tf.image.resize(left, IMG_SIZE) / 255.0
    right = tf.image.resize(right, IMG_SIZE) / 255.0

    if augment:
        left = augment_image(left)
        right = augment_image(right)

    label = tf.convert_to_tensor(sample["diagnosis"], tf.float32)
    return {"left_input": left, "right_input": right}, label


def create_tf_dataset(json_path, shuffle, augment, repeat):
    samples = json.load(open(json_path))

    def gen():
        for s in samples:
            yield s
            if augment:
                yield s

    ds = tf.data.Dataset.from_generator(gen,
        output_signature={"id": tf.TensorSpec([], tf.int32), "leftImage": tf.TensorSpec([], tf.string),
            "rightImage": tf.TensorSpec([], tf.string), "diagnosis": tf.TensorSpec([8], tf.float32), })

    ds = ds.map(lambda x: parse_sample(x, augment), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=800)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds, len(samples) * (2 if augment else 1)


# =====================
# MODEL (STOP AT 512)
# =====================
def create_custom_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for filters in [32, 64, 128, 256, 512]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    return models.Model(inputs, x, name="CNN_Extractor")


def build_model():
    input_shape = (*IMG_SIZE, 3)
    extractor = create_custom_cnn(input_shape)

    left_input = layers.Input(shape=input_shape, name="left_input")
    right_input = layers.Input(shape=input_shape, name="right_input")

    left_feat = extractor(left_input)
    right_feat = extractor(right_input)

    x = layers.Concatenate()([left_feat, right_feat])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.6)(x)

    outputs = layers.Dense(8, activation="sigmoid")(x)

    return models.Model(inputs={"left_input": left_input, "right_input": right_input}, outputs=outputs)


# =====================
# TRAINING
# =====================
if __name__ == "__main__":
    prepare()

    train_ds, train_len = create_tf_dataset("/home/train_dataset.json", shuffle=True, augment=True, repeat=True)
    test_ds, test_len = create_tf_dataset("/home/test_dataset.json", shuffle=False, augment=False, repeat=False)

    model = build_model()

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(multi_label=True, num_labels=8, name="auc_per_class")])

    model.fit(train_ds, validation_data=test_ds, steps_per_epoch=train_len // BATCH_SIZE,
        validation_steps=test_len // BATCH_SIZE, epochs=50,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc_per_class", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc_per_class", factor=0.5, patience=3, min_lr=1e-6)])

    model.save("/home/final_model.keras")

    # =====================
    # EVALUATION
    # =====================
    y_true, y_pred = [], []

    for inputs, labels in test_ds:
        preds = model.predict(inputs, verbose=0)
        y_true.append(labels.numpy())
        y_pred.append(preds)

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    print("\nPer-disease metrics:")
    for i, name in enumerate(DISEASE_NAMES):
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        ap = average_precision_score(y_true[:, i], y_pred[:, i])
        print(f"{name}: AUC={auc:.3f}, AP={ap:.3f}")

    macro_auc = roc_auc_score(y_true, y_pred, average="macro")
    print(f"\nMacro AUC: {macro_auc:.3f}")
