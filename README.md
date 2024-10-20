# “Preferable single-atom catalysts enabled by natural language processing for room temperature Na-S batteries” Code Supplement 

This is the official code supplement of *Preferable single-atom catalysts enabled by natural language processing for room temperature Na-S batteries*.

The repository includes data from the abstracts of the papers used in this experiment, and code related to modeling and data analysis.



`abs_data/` contains relevant papers crawled from Elsevier.

`24ver_aug_emb_training.ipynb` is used to train our augmented embeddings.

`24ver_aug_emb_comparison_new.ipynb` helps to visualize experiments results.

---

**Setup**
Thanks to [1] outstanding work, we developed the code based on the MatSciBERT environment. Please refer to [this link](https://github.com/M3RG-IITD/MatSciBERT) first to configure the required environment.

In addition to above requirements, please install Seaborn in order to plot the charts.

`pip install seaborn`


- [1] [MatSciBERT: A materials domain language model for text mining and information extraction](https://www.nature.com/articles/s41524-022-00784-w)
---



# The tool of predicting effective SA cataysts for RT Na-S batteries based on LLMs
This project utilizes **llama3.1** and for **GPT-4o** converting Excel tables into embeddings, finding the most similar vectors, performing dimensionality reduction, and using a retrieval-augmented generation (RAG) system.

## Installation

### Software  Installation

To run the program, Ollama and OpenSearch are required on your computer. 

For installing OpenSearch, please refer to the link: [Install OpenSearch using Docker](https://opensearch.org/docs/latest/install-and-configure/install-opensearch/docker/).

For installing Ollama, please refer to the link: [Ollama Installation Guide](https://ollama.com/).


### Python dependencies  Installation
To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```
It is recommended to use Spyder to open and run the code in sections.


## Data and Resources

The necessary datasets and generated files required to run the scripts are available for download via the following link:
- **Download Link for Resources**: [Download resources](https://drive.google.com/drive/folders/1MiYhggWrE7LT9Hs5rsCUqqUEBqEp8GOF?usp=sharing)

This package includes all the datasets and any additional files required for processing. 

After downloading, organize the files into the `./resources` directory. The file structure should look like this:

```
.
├── abs_data/
│   ├── Li-S SA_1.csv
│   ├── Low_relevant_1.csv
│   └── ...
├── resources/
│   ├── 240820-3.1.npy
│   ├── dataset_240820.xlsx
│   └── ...
├── ...
└── README.md
```

## Use OpenAI API 

### text-embedding-3-small
**Text Embedding 3 Small** is OpenAI’s small text embedding model, designed for creating embeddings with 1536 dimensions. This model offers a balance between cost-efficiency and performance, making it a great choice for general-purpose vector search applications.


**GPT-4o** is the flagship model of the OpenAI LLM technology portfolio. The O stands for Omni and isn't just some kind of marketing hyperbole, but rather a reference to the model's multiple modalities for text, vision and audio.
The GPT acronym stands for Generative Pre-Trained Transformer. A transformer model is a foundational element of generative AI, providing a neural network architecture that is able to understand and generate new outputs.


```bash
python textembeddings3_openai.py
```



## Use LLAMA 3.1

Llama 3.1 is the latest generation in Meta's family of open large language models (LLMs). It's basically the Facebook parent company's response to OpenAI's GPT and Google's Gemini—but with one key difference: all the Llama models are freely available for almost anyone to use for research and commercial purposes. 

### Converting Excel Tables to Embeddings
Convert your Excel data into usable embeddings with the following command:

```bash
python transfer_embedding.py
```

This script reads an Excel file and converts the data into embeddings using the llama3.1 model.

The following files will be generated by running the script:

- `240820-3.1.npy`
- `dataset_240820_emb.xlsx`
- `prediction.npy`
- `pred_emb.xlsx`


**Note:** This process may take a significant amount of time. If you prefer, you can skip this step and directly use the pre-converted files for further processing. These files are already available in the `./resources/` directory.

###  Dimensionality Reduction Using t-SNE

Reduce the dimensionality of your data to make it suitable for visualization:

```bash
python tsne.py
```

During this process, the following file will be generated:

- `reduced_2d_array_tsne.xlsx`

This file is the result of the dimensionality reduction step.



To find the vectors closest to a specified target vector from the generated embeddings, run:

```bash
python findtop50.py
```

This script assesses the similarity between vectors and lists the top 50.


### Using the RAG System

Implement the Retrieval-Augmented Generation model on your embeddings:

```bash
python ragexcel.py
```

This script uses the RAG model to enhance the processing capabilities of the language model with the embeddings data.

**Citation**

 If you find the code useful for your research, please consider citing our work:
*Preferable single-atom catalysts enabled by natural language processing for room temperature Na-S batteries*

