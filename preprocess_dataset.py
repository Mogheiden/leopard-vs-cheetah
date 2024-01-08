from skimage import io, img_as_ubyte
from skimage.transform import resize
import os

RAW_TRAINING_DATA_DIR = './dataset/training'
RAW_TESTING_DATA_DIR = './dataset/testing'

TRAINING_DATA_PROCESSED_DIR = './dataset/preprocessed/training'
TESTING_DATA_PROCESSED_DIR = './dataset/preprocessed/testing'

def preprocess_image(img):
    img = resize(img, (512, 512))
    return img_as_ubyte(img)

for class_folder in os.listdir(RAW_TESTING_DATA_DIR):
    for file in os.listdir(f'{RAW_TESTING_DATA_DIR}/{class_folder}'):
        input_path = f'{RAW_TESTING_DATA_DIR}/{class_folder}/{file}'
        input_img = io.imread(input_path)
        output_path = f'{TESTING_DATA_PROCESSED_DIR}/{class_folder}/{file}'
        output_img = preprocess_image(input_img)
        io.imsave(output_path, output_img)

for class_folder in os.listdir(RAW_TRAINING_DATA_DIR):
    for file in os.listdir(f'{RAW_TRAINING_DATA_DIR}/{class_folder}'):
        input_path = f'{RAW_TRAINING_DATA_DIR}/{class_folder}/{file}'
        input_img = io.imread(input_path)
        output_path = f'{TRAINING_DATA_PROCESSED_DIR}/{class_folder}/{file}'
        output_img = preprocess_image(input_img)
        io.imsave(output_path, output_img)