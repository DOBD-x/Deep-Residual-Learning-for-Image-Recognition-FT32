# test_tl.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model
with open('fer_tl.json','r') as f:
    model = model_from_json(f.read())
model.load_weights('fer_tl.weights.h5')
print("Loaded transfer-learning model")

# Load test data
X = np.load('modXtest_tl.npy')   # uint8 RGB
y = np.load('modytest_tl.npy')   # one-hot

# Preprocess (same as training)
Xf = preprocess_input(X.astype('float32'))

# Predict
yhat = model.predict(Xf, verbose=0)
predy = yhat.argmax(axis=1).astype(np.int64)
truey = y.argmax(axis=1).astype(np.int64)

# Accuracy
acc = (predy == truey).mean() * 100.0

# Save for confusion matrix
np.save('predy.npy', predy)
np.save('truey.npy', truey)
print(f"Accuracy on TL test set: {acc:.2f}%")
print("Saved predy.npy and truey.npy for confmatrix.py")
