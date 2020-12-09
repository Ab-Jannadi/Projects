import argparse
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, dtype=tf.float32)
    image /= 255
    image = image.numpy()
    return image
def predict(image_path, model, top_k=3):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)
    pred  = model.predict(image)
    pred  = pred[0]
    indices = np.argpartition(pred, len(pred)-1)[:len(pred)]
    min_probs = pred[indices]
    min_probs_order = np.argsort(min_probs)
    ordered_indices = indices[min_probs_order]
    ordered_indices_rev = [a for a in reversed(ordered_indices)]
    classes_0 = ordered_indices_rev[:top_k]
    classes = [a+1 for a in classes_0]
    probs = pred[ordered_indices_rev][:top_k]
    
    return classes, probs


parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="Image path")
parser.add_argument("model", help="The model")
parser.add_argument("--top_k", "top_k", help="top probabilty")
parser.add_argument("--category_names", "category_names", help="category names")

args = parser.parse_args()

image_path = args.image_path
#im = Image.open(image_path)
model = './{}'.format(args.model)
model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
top_k = int(args.top_k)
category_names = args.category_names

with open( category_names='label_map.json', 'r') as f:
    class_names = json.load(f)

classes, probs = predict(image_path, model, top_k=5)

for c, p in zip(classes, probs):
    print('For {} % it is a {}'.format(int(p*100), class_names[str(c)]))

          
# Source:
#  https://docs.python.org/3/howto/argparse.html#id1