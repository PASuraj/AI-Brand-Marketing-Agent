import logging
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def generate_bert_embeddings(texts):
    """
    Generate embeddings for a list of texts using BERT.
    
    Args:
        texts (list): List of texts to convert to embeddings.
        
    Returns:
        embeddings (list): List of embeddings.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def semantic_search(query, articles):
    """
    Perform semantic search to find the most similar articles to the query.
    
    Args:
        query (str): The query to search for.
        articles (list): List of articles, each containing 'title' and 'body'.
    
    Returns:
        List[Dict]: The most relevant articles based on the query.
    """
    article_titles = [article['title'] for article in articles]
    article_texts = [article['body'] for article in articles]

    # Get embeddings for the query and articles using BERT
    query_embedding = generate_bert_embeddings([query])
    article_embeddings = generate_bert_embeddings(article_titles)

    # Compute cosine similarities between the query and articles
    similarities = cosine_similarity(query_embedding, article_embeddings)[0]

    # Get the top 3 most similar articles
    top_indices = np.argsort(similarities)[::-1][:3]
    top_articles = [{'title': article_titles[i], 'body': article_texts[i], 'similarity': similarities[i]} for i in top_indices]

    return top_articles

def process_articles(articles, query):
    """
    Process the scraped articles for semantic search.
    
    Args:
        articles (list): List of scraped articles.
        query (str): The query for semantic search.
    
    Returns:
        list: Articles with semantic search results.
    """
    # Perform semantic search to find relevant articles
    relevant_articles = semantic_search(query, articles)
    logger.info(f"Found {len(relevant_articles)} relevant articles based on the query.")
    return relevant_articles
