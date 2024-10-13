# -*- coding: utf-8 -*-

from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import ollama
import numpy as np
from numpy.linalg import norm


# Function to normalize values between 0 and 1
def normalize_0_1(values):
    min_val = np.min(values)
    max_val = np.max(values)
    return (values - min_val) / (max_val - min_val)

# Load the prediction embeddings
prediction = np.load('./prediction.npy')

# Load the embeddings and t-SNE reduced arrays
embeddings_array = np.load('./240820-3.1.npy')
reduced_array = np.load('240820-3.1-tsne.npy')

# Initialize empty lists to store the results
top50indices = [] 
file_path_1 = './dataset_240820.xlsx'

# Read the Excel file containing paper titles and abstracts
df_dataset = pd.read_excel(file_path_1)
dataset_papername = df_dataset['Column1'].tolist()  # Paper titles
dataset_paperabs = df_dataset['Column2'].tolist()   # Paper abstracts

# Initialize lists for storing results
papername = []
papernumber = []
papersim = []
paperabs = []
tsnex = []
tsney = []

# For each prediction embedding
for j, predict in enumerate(prediction):
    
    # Calculate cosine similarity between the query embedding and all embeddings
    similarities = cosine_similarity([predict], embeddings_array)[0]
    
    # Normalize the similarity scores between 0 and 1
    similarities_norm = normalize_0_1(similarities) 

    # Find the 100 most similar texts
    most_similar_indices = np.argsort(similarities_norm)[-100:][::-1]
    
    # Append results for each similar paper
    for i in most_similar_indices:
        papername.append(dataset_papername[i])  # Paper title
        paperabs.append(dataset_paperabs[i])    # Paper abstract
        papernumber.append(j)                   # Prediction number
        papersim.append(similarities_norm[i])   # Normalized similarity score
        tsnex.append(reduced_array[i+10][0])    # t-SNE x-coordinate
        tsney.append(reduced_array[i+10][1])    # t-SNE y-coordinate

# Create a DataFrame to store the results
df = pd.DataFrame({
    'papername': papername,               # Paper title
    'papernumber': papernumber,           # Prediction number
    'paper_similarity': papersim,         # Similarity score
    'tsnex': tsnex,                       # t-SNE x-coordinate
    'tsney': tsney,                       # t-SNE y-coordinate
    'paper_abstract': paperabs            # Paper abstract
})

# Save the DataFrame to an Excel file
df.to_excel('pred_top50indices_cos_3.1.xlsx', index=False)
