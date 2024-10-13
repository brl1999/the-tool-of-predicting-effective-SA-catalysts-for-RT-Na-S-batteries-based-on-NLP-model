<<<<<<< HEAD
# “Preferable single-atom catalysts enabled by natural language processing for room temperature Na-S batteries” Code Supplement 

This is the official code supplement of *Preferable single-atom catalysts enabled by natural language processing for room temperature Na-S batteries*.

The repository includes data from the abstracts of the papers used in this experiment, and code related to modeling and data analysis.



`abs_data/` contains relevant papers crawled from Elsevier.

`aug_emb_training.ipynb` is used to train our augmented embeddings.

`aug_emb_comparison_new.ipynb` helps to visualize experiments results.

---

**Setup**
Thanks to [1] outstanding work, we developed the code based on the MatSciBERT environment. Please refer to [this link](https://github.com/M3RG-IITD/MatSciBERT) first to configure the required environment.

In addition to above requirements, please install Seaborn in order to plot the charts.

`pip install seaborn`


- [1] [MatSciBERT: A materials domain language model for text mining and information extraction](https://www.nature.com/articles/s41524-022-00784-w)
---

**Citation**

If you find the code useful for your research, please consider citing our work:
*Preferable single-atom catalysts enabled by natural language processing for room temperature Na-S batteries*
=======
﻿# The tool of predicting effective SA cataysts for RT Na-S batteries based on LLAMA3
This project utilizes **llama3** for converting Excel tables into embeddings, finding the most similar vectors, performing dimensionality reduction, conducting 10-fold cross-validation, and using a retrieval-augmented generation (RAG) system.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Data and Resources

The necessary datasets and generated files required to run the scripts are available for download via the following link:
- **Download Link for Resources**: [Download resources](https://drive.google.com/drive/folders/1MiYhggWrE7LT9Hs5rsCUqqUEBqEp8GOF?usp=sharing)

This package includes all the datasets and any additional files required for processing. Make sure to download and extract the contents into the project directory before running the scripts.

Download the files inside and copy them to the same directory as the code file.

## Use OpenAI text-embedding-3-small
**Text Embedding 3 Small** is OpenAI’s small text embedding model, designed for creating embeddings with 1536 dimensions. This model offers a balance between cost-efficiency and performance, making it a great choice for general-purpose vector search applications.

It is recommended to use Spyder to open and run the code in sections.
```bash
python textembedding3.py
```

## Use OpenAI GPT 4o

**GPT-4o** is the flagship model of the OpenAI LLM technology portfolio. The O stands for Omni and isn't just some kind of marketing hyperbole, but rather a reference to the model's multiple modalities for text, vision and audio.
The GPT-4o model marks a new evolution for the GPT-4 LLM that OpenAI first released in March 2023. This isn't the first update for GPT-4 either, as the model first got a boost in November 2023, with the debut of GPT-4 Turbo. The GPT acronym stands for Generative Pre-Trained Transformer. A transformer model is a foundational element of generative AI, providing a neural network architecture that is able to understand and generate new outputs.


It is recommended to use Spyder to open and run the code in sections.
```bash
python gpt4o.py
```

## Use LLAMA 3.1

Llama 3.1 is the latest generation in Meta's family of open large language models (LLM). It's basically the Facebook parent company's response to OpenAI's GPT and Google's Gemini—but with one key difference: all the Llama models are freely available for almost anyone to use for research and commercial purposes. 

### Converting Excel Tables to Embeddings
Convert your Excel data into usable embeddings with the following command:

```bash
python transfer_embedding.py
```

This script reads an Excel file and converts the data into embeddings using the ollama3 model.

### Finding the Top 50 Most Similar Vectors

To find the vectors closest to a specified target vector from the generated embeddings, run:

```bash
python findtop50.py
```

This script assesses the similarity between vectors and lists the top 50.

###  Dimensionality Reduction Using t-SNE

Reduce the dimensionality of your data to make it suitable for visualization:

```bash
python tsne.py
```

This will apply the t-SNE algorithm to the embeddings to help visualize the data in lower dimensions.

### 10-Fold Validation

Validate the model performance using 10-fold cross-validation by executing:

```bash
python 10fold.py
```
This script splits the data into ten parts, using each in turn for testing and the rest for training.

### Using the RAG System

Implement the Retrieval-Augmented Generation model on your embeddings:

```bash
python ragexcel.py
```

This script uses the RAG model to enhance the processing capabilities of the language model with the embeddings data.

>>>>>>> 1186564 (2024)
