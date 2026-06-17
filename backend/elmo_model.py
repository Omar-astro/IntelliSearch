import os
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from backend.preprocessing import preprocess_elmo

CACHE_FILE = 'data/cache/elmo_cache.npy'

class ELMoModel:

    def __init__(self, csv_path='data/Preprocessed_Articles.csv'):

        self.df = pd.read_csv(csv_path)
        self.df.dropna(inplace=True)
        self.df = self.df[['Text']] 
        self.df['processed'] = self.df['Text'].apply(preprocess_elmo)

        print("\n loading the model...(hang in there)!\n")
        self.elmo = hub.load("https://tfhub.dev/google/elmo/3")

        # load cached embeddings if they exist, otherwise compute and save
        if os.path.exists(CACHE_FILE):
            print("loading the cache..\n")
            self.doc_embeddings = np.load(CACHE_FILE)
        else:
            print("first time setup!\n")
            self.doc_embeddings = self.embed_in_batches(self.df['processed'].tolist())
            np.save(CACHE_FILE, self.doc_embeddings)

    def embed_in_batches(self, texts, batch_size=4):

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            print(f"processing batch {i // batch_size + 1}")
            batch = [text[:500] for text in texts[i:i + batch_size]]
            embeddings = self.elmo.signatures['default'](tf.constant(batch))['default']
            all_embeddings.append(embeddings.numpy())

        return np.vstack(all_embeddings)

    def search(self, query, top_k=10):

        query_embedding = self.embed_in_batches([preprocess_elmo(query)])
        scores = cosine_similarity(query_embedding, self.doc_embeddings).flatten()

        results = self.df.copy()
        results['score'] = scores
        results = results.sort_values(by='score', ascending=False).head(top_k)

        return pd.DataFrame({
            'docno'  : results.index.astype(str),
            'score'   : results['score'].values,
            'snippet' : results['Text'].apply(lambda x: x[:200] + '...').values
        }).reset_index(drop=True)