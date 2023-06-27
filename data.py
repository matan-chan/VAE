from config import batch_size, img_size, dataset_repetitions
import tensorflow as tf
from io import BytesIO
from os import listdir
from PIL import Image
import pandas as pd
import numpy as np
import requests
import torch
import clip
import cv2
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device, jit=True)


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def read_img(image_url):
    success = True
    try:
        img_bytes = requests.get(image_url, timeout=0.5).content
        image = np.array(Image.open(BytesIO(img_bytes)))
    except:
        print('timeout')
        success = False
        return np.zeros((img_size, img_size, 3)), success
    return image, success


def get_image(image_url):
    image, _ = tf.py_function(read_img, [image_url], [tf.float64, tf.bool])
    return preprocess_image(image)


def read_image_from_disk(data):
    try:
        img = tf.image.decode_jpeg(tf.io.read_file(data))
        img = tf.cast(img, tf.float32)
        img = tf.clip_by_value(img / 255.0, 0.0, 1.0)
        return img
    except:
        img = np.zeros((img_size, img_size, 3))
        return tf.cast(img, tf.float32)


def preprocess_image(image, clip=True):
    if len(tf.shape(image)) != 3:
        image = image[..., np.newaxis]
        image = tf.image.grayscale_to_rgb(image)

    height, width, depth = tf.shape(image)
    if depth == 4:
        image = image[:, :, :3]

    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(image, (height - crop_size) // 2, (width - crop_size) // 2, crop_size,
                                          crop_size)
    image = tf.image.resize(image, size=[img_size, img_size], antialias=True)
    if not clip:
        return image
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


def get_dataframe():
    only_files = ['data/meta/' + file for file in listdir('data/meta/')]
    engine = ['pyarrow'] * len(only_files)
    data_frame = pd.concat(map(pd.read_parquet, only_files, engine))
    return data_frame


def extract_relevant_data_from_files():
    data_frame = get_dataframe()
    return data_frame['url']


def download_images(start_in_batch=778396, run_for=10, start_name=0):
    urls = extract_relevant_data_from_files()

    for i, url in enumerate(urls):
        try:
            # if i - start_in_batch == run_for:
            #     break
            if i > start_in_batch:

                image, success = read_img(url)
                if success:
                    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    im_rgb = np.array(preprocess_image(im_rgb, False))
                    cv2.imwrite(f'data/images/{i + start_name}.jpg', im_rgb)
        except:
            continue


def prepare_dataset(downloaded=False):
    if downloaded:
        list_ds = tf.data.Dataset.list_files(r"data/images/*", shuffle=True). \
            shuffle(10 * batch_size).repeat(dataset_repetitions).batch(batch_size, drop_remainder=True).prefetch(
            buffer_size=tf.data.AUTOTUNE)

        return map(lambda x: np.array(list(map(read_image_from_disk, x))), list_ds)
    else:
        urls = extract_relevant_data_from_files()
        list_ds = tf.data.Dataset.from_tensor_slices(urls). \
            shuffle(10 * batch_size).repeat(dataset_repetitions).batch(batch_size, drop_remainder=True).prefetch(
            buffer_size=tf.data.AUTOTUNE)  # cache().
        return map(lambda x: np.array(list(map(get_image, x))), list_ds)



