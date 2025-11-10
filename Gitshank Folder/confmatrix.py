import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# --- Data Loading ---
try:
    y_true = np.load('./truey.npy')
    y_pred = np.load('./predy.npy')
except FileNotFoundError:
    print("Error: Could not load 'truey.npy' or 'predy.npy'. Please ensure they are in the correct location.")
    exit()

cm = confusion_matrix(y_true, y_pred)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
title='Confusion matrix'

# --- Normalization Step ---
# Normalize the confusion matrix: divide each row by the sum of that row.
# This makes the sum of values in each True Label row equal to 1.0 (or 100%).
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# --- Plotting the Normalized Matrix ---
plt.figure(figsize=(8, 6)) # Optional: makes the plot larger

# Plot the normalized matrix
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
fmt = '.2f'
thresh = cm_normalized.max() / 2.

for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
    plt.text(j, i, format(cm_normalized[i, j], fmt),
             horizontalalignment="center",
             # Use the normalized matrix for color decision
             color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
# plt.show()
plt.savefig('confusion_matrix.png')
print("Confusion matrix plot saved as confusion_matrix.png")