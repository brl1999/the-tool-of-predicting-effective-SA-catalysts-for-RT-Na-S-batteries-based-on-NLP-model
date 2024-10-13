import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

# Load .npy files
array1 = np.load('./prediction.npy')
array2 = np.load('./240820.npy')

# Concatenate two arrays, assuming concatenation along rows (i.e., vertical merge)
combined_array = np.concatenate((array1, array2), axis=0)

# Check the shape of the combined array
print(f"Combined array shape: {combined_array.shape}")

# Use t-SNE to reduce dimensionality from 4096 dimensions to 3 dimensions
tsne = TSNE(n_components=3, random_state=42)
reduced_array = tsne.fit_transform(combined_array)

# Check the shape of the reduced array
print(f"Reduced array shape: {reduced_array.shape}")

# Save the reduced data as an Excel file
df = pd.DataFrame(reduced_array, columns=['Dim1', 'Dim2', 'Dim3'])
df.to_excel('reduced_3d_array_tsne.xlsx', index=False)

# Print some data to verify
print(df.head())
