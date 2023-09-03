from __future__ import print_function
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')
import pickle

# Load the trained model
RF_pkl_filename = 'RandomForest.pkl'
RF_Model_pkl = open(RF_pkl_filename, 'rb')
RF = pickle.load(RF_Model_pkl)

# Load the crop recommendation dataset
PATH = 'Crop_recommendation.csv'
df = pd.read_csv(PATH)

# Define the crop list
crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']

# Define input values
input_values = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])

# Make prediction and get probabilities
prediction = RF.predict(input_values)
probabilities = RF.predict_proba(input_values)[0]

# Create a list of recommended crops in order of decreasing probability
recommended_crops = []
for i in np.argsort(probabilities)[::-1]:
    if probabilities[i] > 0:
        recommended_crops.append((crops[i], probabilities[i]))

# Output the recommended crops with their corresponding probabilities
print("Based on the input values, we recommend the following crops (in order of best to worst):")
for crop, probability in recommended_crops:
    print(f"- {crop} ({probability:.4f})")
