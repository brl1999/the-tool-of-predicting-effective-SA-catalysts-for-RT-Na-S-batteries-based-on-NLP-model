# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import cross_validate

# # Assume the data is already loaded into a DataFrame
# # For example, using the pd.read_excel() method to load data from a file
file_path = 'ceshiji3_emb.xlsx'  # Replace with your Excel file path
data = pd.read_excel(file_path)

embeddings = np.load('./ceshiji3.npy')

file_path = 'ceshiji3.xlsx'  # Replace with your Excel file path
data = pd.read_excel(file_path)

labels = data['Label'].values

# Ensure that the number of embeddings matches the number of labels
if len(embeddings) != len(labels):
    print(f"Mismatch: {len(embeddings)} embeddings and {len(labels)} labels")
else:
    print("Embeddings and labels match in size. Proceeding with model training...")
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Define MLP model
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, learning_rate_init=0.01,random_state=42)
    
    # Perform 10-fold cross-validation
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'f1': make_scorer(f1_score, average='weighted'),
        'roc_auc': make_scorer(roc_auc_score, multi_class='ovo', needs_proba=True),
        'average_precision': make_scorer(average_precision_score, average='weighted', needs_proba=True)
    }

    # Use cross_validate for 10-fold cross-validation and get these metrics
    results = cross_validate(mlp, embeddings_scaled, labels, cv=3, scoring=scoring)

    # Display the average score for each metric
    print("Accuracy (average):", np.mean(results['test_accuracy']))
    print("Precision (average):", np.mean(results['test_precision']))
    print("F1-Score (average):", np.mean(results['test_f1']))
    print("AUROC (average):", np.mean(results['test_roc_auc']))
    print("AUPR (average):", np.mean(results['test_average_precision']))
