# Converts grayscale to RGB, upscales to 224×224, one-hot encodes labels.
import pandas as pd
import numpy as np
import cv2

IMG_SIZE = 224  # For EfficientNet/ResNet
WIDTH = HEIGHT = 48

data = pd.read_csv('./fer2013.csv')
pixels = data['pixels'].tolist()

X = []
for p in pixels:
    arr = np.asarray([int(px) for px in p.split(' ')], dtype=np.uint8).reshape(HEIGHT, WIDTH)  # (48,48)
    rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)                       # -> (48,48,3)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)  # -> (224,224,3)
    X.append(rgb)

X = np.asarray(X, dtype=np.uint8)  # keep as uint8; we'll apply model-specific preprocess later

# One-hot labels (0..6)
y = pd.get_dummies(data['emotion']).values.astype('float32')

np.save('fdataX_tl.npy', X)
np.save('flabels_tl.npy', y)

print("Transfer-learning preprocessing done.")
print("X:", X.shape, " y:", y.shape)
