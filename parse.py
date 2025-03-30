import logging
import faiss
import numpy as np
import torch
import pickle
import os
from transformers import BertTokenizer, BertModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load BERT model and tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
except Exception as e:
    logger.error(f"Error loading BERT model: {e}")
    raise

# FAISS index and metadata files
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

# Load or initialize FAISS index
def load_faiss_index():
    try:
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(METADATA_FILE):
            index = faiss.read_index(FAISS_INDEX_FILE)
            with open(METADATA_FILE, "rb") as f:
                metadata = pickle.load(f)  # Load article metadata
            logger.info("FAISS index loaded from disk.")
        else:
            index = faiss.IndexFlatL2(768)  # L2 distance for 768-dim vectors
            metadata = []  # Empty metadata list
            logger.info("Initialized new FAISS index.")
        return index, metadata
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        raise

# Save FAISS index & metadata
def save_faiss_index(index, metadata):
    try:
        faiss.write_index(index, FAISS_INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(metadata, f)  # Save metadata
        logger.info("FAISS index saved to disk.")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")
        raise

# Generate BERT embeddings
def generate_bert_embeddings(texts):
    try:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()  # Extract CLS token embeddings
    except Exception as e:
        logger.error(f"Error generating BERT embeddings: {e}")
        raise

# Store embeddings in FAISS
def store_embeddings(articles):
    try:
        index, metadata = load_faiss_index()

        article_titles = [article["title"] for article in articles]
        article_bodies = [article["body"] for article in articles]

        embeddings = generate_bert_embeddings(article_bodies)

        index.add(embeddings)  # Add new embeddings to FAISS
        metadata.extend(zip(article_titles, article_bodies))  # Store metadata

        save_faiss_index(index, metadata)
        logger.info(f"Stored {len(articles)} new articles in FAISS.")
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
        raise

# Perform similarity search
def semantic_search(query, top_k=3):
    try:
        index, metadata = load_faiss_index()

        if index.ntotal == 0:
            logger.warning("FAISS index is empty.")
            return []

        query_embedding = generate_bert_embeddings([query])
        distances, indices = index.search(query_embedding, top_k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(metadata):
                title, body = metadata[idx]
                results.append({"title": title, "body": body, "distance": distances[0][i]})
        
        return results
    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
        raise

# Process new articles: store, search, and pass to LLM
def process_articles(articles):
    try:
        index, metadata = load_faiss_index()
        
        results = []
        
        for article in articles:
            title, body = article["title"], article["body"]
            logger.info(f"Processing article: {title}")

            # Find similar articles
            similar_articles = semantic_search(title)

            # Save new article in FAISS
            store_embeddings([article])

            # Prepare input for LLM: new article + similar historical articles
            llm_input = {
                "new_article": {"title": title, "body": body},
                "similar_articles": similar_articles
            }       
            results.append(llm_input)
        
        return results
    except Exception as e:
        logger.error(f"Error processing articles: {e}")
        raise
