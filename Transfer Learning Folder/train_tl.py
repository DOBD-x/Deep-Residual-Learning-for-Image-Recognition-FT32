# train_tl.py
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from collections import Counter
import matplotlib.pyplot as plt

NUM_CLASSES = 7
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS_WARMUP = 8     # train head only
EPOCHS_FINETUNE = 25  # fine-tune top of the backbone
LABEL_SMOOTH = 0.1

# --- Load data ---
X = np.load('fdataX_tl.npy')       # uint8, shape (N,224,224,3)
y = np.load('flabels_tl.npy')      # one-hot

# Split: test 10%, val 10% of the remainder
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y.argmax(1))
X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.10, random_state=41, stratify=y_train.argmax(1))

np.save('modXtest_tl.npy', X_test)
np.save('modytest_tl.npy', y_test)

# --- Data augmentation pipeline ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="aug")

# --- Preprocess for EfficientNet ---
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

def make_ds(X, y, training):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(8192, reshuffle_each_iteration=True)
        ds = ds.map(lambda a,b: (data_augmentation(tf.cast(a, tf.float32)), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda a,b: (tf.cast(a, tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE)
    # model-specific normalization
    ds = ds.map(lambda a,b: (preprocess_input(a), b), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds(X_train, y_train, True)
val_ds   = make_ds(X_val,   y_val,   False)

# --- Build model ---
base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False  # warm-up head first

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = inputs
x = base(x, training=False)   # important for BN in frozen backbone
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
              metrics=['accuracy'])

# --- Class weights to handle imbalance ---
counts = Counter(y_train.argmax(1))
total = sum(counts.values())
class_weight = {cls: total/(NUM_CLASSES*count) for cls, count in counts.items()}

# --- Callbacks ---
cbs = [
    EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6, monitor='val_loss'),
    ModelCheckpoint('fer_tl_best.h5', save_best_only=True, monitor='val_accuracy')
]

# --- Warm-up: train head only ---
hist1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_WARMUP,
                  class_weight=class_weight, callbacks=cbs, verbose=1)

# --- Fine-tune: unfreeze top of backbone ---
# Unfreeze top 30% of layers (heuristic)
num_layers = len(base.layers)
for i, layer in enumerate(base.layers):
    if i >= int(num_layers*0.70):
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
              metrics=['accuracy'])

hist2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
                  class_weight=class_weight, callbacks=cbs, verbose=1)

# --- Save as JSON + weights (like your original flow) ---
fer_json = model.to_json()
with open('fer_tl.json', 'w') as f:
    f.write(fer_json)
model.save_weights('fer_tl.weights.h5')
print("Saved transfer-learning model to disk")

# --- Plot and save training curves ---
def _concat_hist(h1, h2):
    hist = {}
    for k in h1.history.keys():
        hist[k] = h1.history[k] + (h2.history.get(k, []) if h2 else [])
    return hist

history = _concat_hist(hist1, hist2)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history['accuracy'], label='Train Acc')
plt.plot(history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.savefig('training_history_tl.png')
print("Saved training_history_tl.png")
