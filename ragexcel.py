# -*- coding: utf-8 -*-


import ollama
import time
import os
import json
import numpy as np
from numpy.linalg import norm
import pandas as pd


# Open a file and return paragraphs
def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs


def save_embeddings(filename, embeddings):
    # Create directory if it doesn't exist
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    # Dump embeddings to json
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)


def load_embeddings(filename):
    # Check if file exists
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    # Load embeddings from json
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)


def get_embeddings(filename, modelname, chunks):
    # Check if embeddings are already saved
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    # Get embeddings from ollama
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    # Save embeddings
    save_embeddings(filename, embeddings)
    return embeddings


# Find cosine similarity of every chunk to a given embedding
def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)


# Assuming the file is named "data.xlsx", modify based on the actual file name
def write_paragraph(file_path):
    # Read the Excel file
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Print column names to check if correct
    print("Columns in the Excel file:", df.columns)

    paragraphs = []
    
    # Check if the required columns exist
    if 'Column1' not in df.columns or 'Column2' not in df.columns:
        print("Error: 'Column1' or 'Column2' not found in the Excel file.")
        return

    # Iterate through each row and merge Column1 and Column2
    for index, row in df.iterrows():
        paragraph = f"Title: {row['Column1']} Abstract: {row['Column2']}\n"  # Merge columns and add a newline
        paragraphs.append(paragraph)
    
    return paragraphs


def main():
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context.The context below are the abstract and title of papers. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know. These are the title and the abstract of the paper
        Context:
    """
    # Open file
    filename = "output.txt"
    #paragraphs = parse_file(filename)
    file_path = "./resources/dataset_240820.xlsx"
    
    paragraphs = write_paragraph(file_path)

    # embeddings = get_embeddings(filename, "llama3", paragraphs)
    # print(np.array(embeddings).shape)
    embeddings = np.load('./resources/240820-3.1.npy')
    
    prompt = input("What do you want to know? -> ")
    # Strongly recommended that all embeddings are generated by the same model (don't mix and match)
    prompt_embedding = ollama.embeddings(model="llama3", prompt=prompt)["embedding"]
    # Find most similar to each other
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:100]

     
    df = pd.DataFrame(columns=["Iteration", "Response", "Title and Abstract:"])

    for i in range(100):
        # Call ollama.chat to generate response
        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                    + "\n".join(paragraphs[most_similar_chunks[i][1]]),
                },
                {"role": "user", "content": prompt},
            ],
        )

        # Get the generated content and write it to DataFrame with the iteration number
        response_content = response["message"]["content"]
        new_row = pd.DataFrame({
            "Iteration": [i+1], 
            "Response": [response_content], 
            "Title and Abstract": [paragraphs[most_similar_chunks[i][1]]]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        
        

    # Save the results to an Excel file
    df.to_excel("output_responses.xlsx", index=False)
    print("Responses written to 'output_responses.xlsx'.")

if __name__ == "__main__":
    main()
