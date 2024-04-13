"""#RAG from scratch (Retrieval Augmented Generation)
#It takes info, and passes it to an llm, and the llm handles the response
#-Retrieval -Find info on query
#Augmented - We take retrived info, and augment it into our prompt
#Generation - generates info on our given augmented prompt
# main goal is to improve generation of the output
#------Uses-------
#Prevents Hallucinations
#Adds access to private info (custom/specific) after model creation"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import numpy as np
from daft import DataFrame
from tqdm.auto import tqdm
import typing
import pandas as pd
from spacy.lang.en import English
import time
import re
from sentence_transformers import SentenceTransformer


def main():
    chunk_size = 15
    min_token_len = 30
    embedding_batch_size = 10
    ReEmbed = True
    # loading text data
    scripts = []
    for filename in tqdm(os.listdir('./data'), desc='Loading data'):
        if filename.endswith('.txt'):
            filename = './data/' + filename
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
                d = {'title': filename[7:-4],
                     'num_words': len(text.split()),
                     'num_characters': len(text),
                     'num_tokens': len(text) // 4,
                     'text': text}
                scripts.append(d)

    # embeddings management (to truncate data into chunks that the llm can accept)
    # Add a pipeline to create sentences (https://spacy.io/api/sentencizer)
    nlp = English()
    nlp.add_pipe('sentencizer')

    # Create a doc instance
    for item in tqdm(scripts, desc='Sentencing'):
        item['sentences'] = list(nlp(item['text']).sents)
        item['sentences'] = [str(sentence) for sentence in item['sentences']]
        item['num_sentences'] = len(item['sentences'])

    # Chunking (can be done using langchain)
    # Done so that embedding can be done easily and so that the llm can be more specific and focused



    def chunker(input_list: list[str], chunk_size: int) -> list[list[str]]:
        return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


    for item in tqdm(scripts, desc='Chunking'):
        item['chunks'] = chunker(item['sentences'], chunk_size=chunk_size)
        item['num_chunks'] = len(item['chunks'])

    # Adding chunks as seperate field
    scripts_with_chunks = []
    for item in tqdm(scripts, desc='Adding chunk metadata'):
        for chunk in item['chunks']:
            chunk_dict = {}

            # join list of sentences in a chunk back into one para
            joined_chunk = ''.join(chunk).replace("  ", " ").strip()
            joined_chunk = re.sub('\n+', '\n\n', joined_chunk)
            joined_chunk = re.sub('\s+', ' ', joined_chunk)
            # joined sentence chunk
            chunk_dict['title'] = item['title']
            chunk_dict['chunks'] = item['title']+": " + joined_chunk

            chunk_dict['chunk_char_count'] = len(joined_chunk)
            chunk_dict['chunk_word_count'] = len([word for word in joined_chunk.split()])
            chunk_dict['chunk_token_count'] = len(joined_chunk) // 4

            scripts_with_chunks.append(chunk_dict)
    df = pd.DataFrame(scripts_with_chunks)

    scripts_with_chunks_over_min_token_length = df[df['chunk_token_count'] > min_token_len].to_dict(orient='records')

    # Embedding
    embedding_model = SentenceTransformer('nomic-embed-text-v1', trust_remote_code=True)
    embedding_model.to('cuda')

    embeddings_df_save_path = 'text_chunks_and_embeddings_df.csv'
    if ReEmbed:
        for item in tqdm(scripts_with_chunks_over_min_token_length):
            item["embedding"] = embedding_model.encode(item["chunks"])
        text_chunks_and_embeddings_df = pd.DataFrame(scripts_with_chunks_over_min_token_length)
        text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)


    """if ReEmbed:
        text_chunks = [item["chunks"] for item in scripts_with_chunks_over_min_token_length]
    
        text_chunk_embeddings = embedding_model.encode(text_chunks, batch_size=embedding_batch_size, convert_to_tensor=True)
        item["embedding"] = text_chunk_embeddings
        text_chunks_and_embeddings_df = pd.DataFrame(scripts_with_chunks_over_min_token_length)
        text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)
    else:
        text_chunks_and_embeddings_df_load = pd.read_csv(embeddings_df_save_path)"""

if __name__ == "__main__":
    main()