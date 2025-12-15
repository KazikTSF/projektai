import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

BATCH_SIZE = 32
IMG_SIZE = (512, 512)


def prepare(test_size=0.2, random_state=42):
    df = pd.read_csv("/kaggle/input/ocular-disease-recognition-odir5k/full_df.csv", sep=',')

    samples = []
    for _, row in df.iterrows():
        diagnosis_vector = [int(row[c]) for c in ["N", "D", "G", "C", "A", "H", "M", "O"]]
        samples.append({
            "id": int(row["ID"]),
            "leftImage": f"/home/processed/{row['ID']}_left.jpg",
            "rightImage": f"/home/processed/{row['ID']}_right.jpg",
            "diagnosis": diagnosis_vector
        })

    train_samples, test_samples = train_test_split(
        samples,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    with open("/home/train_dataset.json", "w") as f:
        json.dump(train_samples, f, indent=4)
    with open("/home/test_dataset.json", "w") as f:
        json.dump(test_samples, f, indent=4)

    return len(train_samples), len(test_samples)


def augment_image(image):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE[0], IMG_SIZE[1])
    image = tf.image.random_crop(image, size=[IMG_SIZE[0], IMG_SIZE[1], 3])
    return image


def parse_sample(sample, augment):
    left = tf.io.read_file(sample["leftImage"])
    left = tf.image.decode_jpeg(left, channels=3)
    left = tf.image.resize(left, IMG_SIZE)
    left = left / 255.0

    right = tf.io.read_file(sample["rightImage"])
    right = tf.image.decode_jpeg(right, channels=3)
    right = tf.image.resize(right, IMG_SIZE)
    right = right / 255.0

    if augment:
        left = augment_image(left)
        right = augment_image(right)

    diagnosis = tf.convert_to_tensor(sample["diagnosis"], dtype=tf.float32)
    return {"left_input": left, "right_input": right}, diagnosis


def create_tf_dataset(json_path, shuffle, augment, repeat):
    with open(json_path, "r") as f:
        samples = json.load(f)

    if augment:
        samples = samples * 2

    def gen():
        for sample in samples:
            yield sample

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "id": tf.TensorSpec(shape=(), dtype=tf.int32),
            "leftImage": tf.TensorSpec(shape=(), dtype=tf.string),
            "rightImage": tf.TensorSpec(shape=(), dtype=tf.string),
            "diagnosis": tf.TensorSpec(shape=(8,), dtype=tf.float32),
        }
    )

    dataset = dataset.map(lambda x: parse_sample(x, augment), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=500)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset, len(samples)


def create_custom_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)

    return models.Model(inputs=inputs, outputs=x, name="Custom_CNN_Extractor")


def build_model(num_classes=8):
    input_shape = (*IMG_SIZE, 3)

    feature_extractor = create_custom_cnn(input_shape)

    left_input = layers.Input(shape=input_shape, name="left_input")
    right_input = layers.Input(shape=input_shape, name="right_input")

    left_features = feature_extractor(left_input)
    right_features = feature_extractor(right_input)

    combined = layers.Concatenate()([left_features, right_features])

    combined = layers.Dense(256, activation="relu")(combined)
    combined = layers.Dropout(0.5)(combined)

    outputs = layers.Dense(num_classes, activation="sigmoid")(combined)

    model = models.Model(inputs={"left_input": left_input, "right_input": right_input}, outputs=outputs)
    return model


if __name__ == "__main__":
    train_count, test_count = prepare()
    train_dataset, train_len = create_tf_dataset("/home/train_dataset.json", shuffle=True, augment=True, repeat=True)
    test_dataset, test_len = create_tf_dataset("/home/test_dataset.json", shuffle=False, augment=False, repeat=False)

    steps_per_epoch = train_len // BATCH_SIZE
    validation_steps = test_len // BATCH_SIZE

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("/home/best_model.keras", monitor="val_auc", mode="max",
                                           save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, min_lr=1e-6)
    ]

    model.fit(
        train_dataset,
        validation_data=test_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=50,
        callbacks=callbacks
    )

    model.save("/home/final_model.keras")