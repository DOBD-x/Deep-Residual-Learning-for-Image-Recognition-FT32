# Deep-Residual-Learning-for-Image-Recognition-FT32
For Facial Emotion Recognition (FER) model
==============================================================
Create a folder ‘fer2013’ in your home directory
1. Place the following files into the folder; they will be used to run gitshank’s model:
- fer2013.csv
- preprocessing.py
- fertrain.py
- fertest.py
- confmatrix.py
- recognize2.py

2. In your IDE’s terminal run the following commands after navigating to the fer2013 folder:
   
cd fer2013


3. Then Set Up a Virtual Environment:

python -m venv venv
.\venv\Scripts\activate


4. Install Required Libraries inside the activated environment:

pip install pytorch
pip install tensorflow keras numpy scikit-learn pandas opencv-python matplotlib


5. Run Preprocessing:

python preprocessing.py

- This will parse fer2013.csv into usable features and labels
- They will be saved as:
   - fdataX.npy → input features
   - flabels.npy → one-hot encoded emotion labels


6. Train the Model:

python fertrain.py

This will generate:
- fer.weights.h5 → trained model weights
- fer.json → model architecture
- modXtest.npy, modytest.npy → test data
- training_history.png → training history


7. Test Accuracy with fertest.py:

python fertest.py

- This script loads your trained model and test data (modXtest.npy, modytest.npy) to compute accuracy.


8. Generate Confusion Matrix with:
   
python confmatrix.py

- This script shows which emotions your model confuses most often.


9. Run recognize2.py

- This script is an improvement to the algorithm that improves the previous data set. Expected improvement is 80% from 71.55%
