import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumour10Epochs.h5')

img = cv2.imread('pred/pred2.jpg')
img = Image.fromarray(img)
img = img.resize((64, 64))


img = np.array(img)
input_img = np.expand_dims(img, axis=0)

result = model.predict(input_img)
print(result)