# -*- coding: utf-8 -*-


import pandas as pd
import ollama
import numpy as np

#transfer dataset_240820
# Read .xlsx file
df = pd.read_excel('./resources/dataset_240820.xlsx')
print(df.columns)

# Assume column 'Abstract' contains the text to be converted
texts = df['Column2'].tolist()

# Initialize an empty list to store the embedding vectors
embeddings = []

for i, text in enumerate(texts):
    response = ollama.embeddings(model="llama3.1", prompt=text)
    embedding = response["embedding"]
    embeddings.append(embedding)
    print('Transfer ' + str(i) + '/' + str(len(texts)))

# Convert the list of embeddings to a numpy array
embeddings_array = np.array(embeddings)
np.save('./resources/240820-3.1.npy', embeddings_array)

print(embeddings_array.shape)

# Convert the embeddings to strings and add them to the DataFrame
embeddings_str = [' '.join(map(str, embedding)) for embedding in embeddings]

# Add a new column to store the embeddings
df['Embedding'] = embeddings_str

# Save the modified DataFrame to a new .xlsx file
df.to_excel('dataset_240820_emb.xlsx', index=False)

print("Embedding vectors have been successfully added to the DataFrame and saved to a new .xlsx file.")


#%%    
#transfer prediction 
# Read .xlsx file
df = pd.read_excel('./resources/prediction dataset.xlsx')
print(df.columns)

# Assume column 'Abstract' contains the text to be converted
texts = df['Column2'].tolist()

# Initialize an empty list to store the embedding vectors
embeddings = []

for i, text in enumerate(texts):
    response = ollama.embeddings(model="llama3.1", prompt=text)
    embedding = response["embedding"]
    embeddings.append(embedding)
    print('Transfer ' + str(i) + '/' + str(len(texts)))

# Convert the list of embeddings to a numpy array
embeddings_array = np.array(embeddings)
np.save('./resources/prediction.npy', embeddings_array)

print(embeddings_array.shape)

# Convert the embeddings to strings and add them to the DataFrame
embeddings_str = [' '.join(map(str, embedding)) for embedding in embeddings]

# Add a new column to store the embeddings
df['Embedding'] = embeddings_str

# Save the modified DataFrame to a new .xlsx file
df.to_excel('pred_emb.xlsx', index=False)

print("Embedding vectors have been successfully added to the DataFrame and saved to a new .xlsx file.")



#%%    
#transfer ceshiji3
# Read .xlsx file
df = pd.read_excel('./resources/ceshiji3.xlsx')
print(df.columns)

# Assume column 'Abstract' contains the text to be converted
texts = df['Abstract'].tolist()

# Initialize an empty list to store the embedding vectors
embeddings = []

for i, text in enumerate(texts):
    response = ollama.embeddings(model="llama3.1", prompt=text)
    embedding = response["embedding"]
    embeddings.append(embedding)
    print('Transfer ' + str(i) + '/' + str(len(texts)))

# Convert the list of embeddings to a numpy array
embeddings_array = np.array(embeddings)
np.save('./resources/ceshiji3.npy', embeddings_array)

print(embeddings_array.shape)

# Convert the embeddings to strings and add them to the DataFrame
embeddings_str = [' '.join(map(str, embedding)) for embedding in embeddings]

# Add a new column to store the embeddings
df['Embedding'] = embeddings_str

# Save the modified DataFrame to a new .xlsx file
df.to_excel('ceshiji3_emb.xlsx', index=False)

print("Embedding vectors have been successfully added to the DataFrame and saved to a new .xlsx file.")