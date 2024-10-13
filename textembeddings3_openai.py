# -*- coding: utf-8 -*-


import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings

df_all = pd.read_excel(r".\dataset_240820.xlsx", header=None).drop_duplicates()
df_pre = pd.read_excel(r".\prediction dataset.xlsx").drop_duplicates()

embedding_model = "text-embedding-3-small"
openai_api_key = "your API keys"
embeddings = OpenAIEmbeddings(model = embedding_model, openai_api_key = openai_api_key)

texts = df_all.iloc[:, 1].tolist()  # Abstract -> Embedding (vectorize)
# metadata for output: title, abstract, journal name
metadata = df_all.iloc[:, [0, 1, 2]].rename(columns={df_all.columns[0]: 'TITLE', df_all.columns[1]: 'ABSTRACT', df_all.columns[2]: 'JOURNAL_NAME'}).to_dict(orient='records')

#%%    
from langchain.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
opensearch_url = "https://localhost:9200"
opensearch_client = OpenSearch(
    opensearch_url,  
    use_ssl=True,             
    verify_certs=False,        
    http_auth=('admin', 'StrongPass#123'),  
    timeout=120                
)

vector_store =OpenSearchVectorSearch.from_texts(
    http_auth=('admin', 'StrongPass#123'),
    texts=texts,
    embedding=embeddings,
    use_ssl=True,
    verify_certs=False,
    opensearch_url = opensearch_url,
    metadatas=metadata,
    index_name="dataset_240820",
    bulk_size=12000
)


#%% 
query = df_pre.iloc[0,0] +" "+ df_pre.iloc[0,1]

results = vector_store.similarity_search_with_score(query, k=50, space_type = 'cosinesimil')

#%% 

data = []
for result, score in results:
    data.append({
        'Title': result.metadata.get('TITLE', ''),
        'Abstract': result.page_content,
        'Journal_name': result.metadata.get('JOURNAL_NAME', ''),
        'Cosinesimil_Score': score
    })
    
df_results = pd.DataFrame(data)

df_results = pd.concat([pd.DataFrame([{'Title': df_pre.iloc[0,0], 'Abstract': df_pre.iloc[0,1]}]), df_results], ignore_index=True)

dfs_results = []

# Loop 
for i in range(df_pre.shape[0]):
    # query = df_pre.iloc[i, 0] + " " + df_pre.iloc[i, 1]
    query = df_pre.iloc[i, 1]
    results = vector_store.similarity_search_with_score(query, k=100, space_type='cosinesimil')

    data = []
    for result, score in results:
        data.append({
            'Title': result.metadata.get('TITLE', ''),
            'Abstract': result.page_content,
            'Journal_name': result.metadata.get('JOURNAL_NAME', ''),
            'Cosinesimil_Score': score
        })
    
    df_results = pd.DataFrame(data)
    df_results = pd.concat([pd.DataFrame([{'Title': df_pre.iloc[i, 0], 'Abstract': df_pre.iloc[i, 1]}]), df_results], ignore_index=True)
    
    dfs_results.append(df_results)
    
output_name = 'output_multiple_sheets_top100_abs.xlsx'

with pd.ExcelWriter(output_name) as writer:
    for i, df in enumerate(dfs_results):
        sheet_name = f'Sheet{i+1}'
        
        df.to_excel(writer, index=False, sheet_name=sheet_name)
