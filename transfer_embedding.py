# -*- coding: utf-8 -*-


import pandas as pd
import ollama
import numpy as np

# Read .xlsx file
df = pd.read_excel('ceshiji3.xlsx')
print(df.columns)

# Assume column 'Abstract' contains the text to be converted
texts = df['Abstract'].tolist()

# Initialize an empty list to store the embedding vectors
embeddings = []

for i, text in enumerate(texts):
    response = ollama.embeddings(model="llama3", prompt=text)
    embedding = response["embedding"]
    embeddings.append(embedding)
    print('Transfer ' + str(i) + '/' + str(len(texts)))

# Convert the list of embeddings to a numpy array
embeddings_array = np.array(embeddings)
np.save('ceshiji3.npy', embeddings_array)

print(embeddings_array.shape)

# Convert the embeddings to strings and add them to the DataFrame
embeddings_str = [' '.join(map(str, embedding)) for embedding in embeddings]

# Add a new column to store the embeddings
df['Embedding'] = embeddings_str

# Save the modified DataFrame to a new .xlsx file
df.to_excel('dataset_240820_ceshiji3.xlsx', index=False)

print("Embedding vectors have been successfully added to the DataFrame and saved to a new .xlsx file.")
