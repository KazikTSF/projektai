import pandas as pd
import json
import tensorflow as tf

def prepare():
    df = pd.read_csv("data/data.csv", sep=';')

    samples = []
    for _, row in df.iterrows():
        sample = {
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
        }
        samples.append(sample)
    with open("data/prepared_dataset.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_crop(image, size=[int(image.shape[0]*0.9), int(image.shape[1]*0.9), 3])
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

    diagnosis = tf.convert_to_tensor(sample["diagnosis"], dtype=tf.int32)
    return {"left_input": left, "right_input": right}, diagnosis

def create_tf_dataset(batch_size=32, shuffle=True, augment=False, img_size=(224,224)):
    with open("data/prepared_dataset.json", "r") as f:
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

    dataset = dataset.map(lambda x: parse_sample(x, img_size, augment), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(samples))

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    prepare()
    train_dataset = create_tf_dataset(batch_size=32, augment=True)

