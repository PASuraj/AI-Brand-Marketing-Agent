from sentence_transformers import SentenceTransformer, util
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SemanticProcessor:
    """Handle semantic search and content processing."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with specified embedding model."""
        self.model = SentenceTransformer(model_name)
        
    def search_articles(
        self, 
        articles: List[Dict], 
        query: str, 
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Search articles using semantic similarity.
        
        Args:
            articles: List of article dictionaries
            query: Search query
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching articles with similarity scores
        """
        try:
            # Encode search query
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            matches = []
            
            # Process each article
            for article in articles:
                # Combine title and body for context
                content = f"{article['title']} {article['body']}"
                content_embedding = self.model.encode(content, convert_to_tensor=True)
                
                # Calculate similarity
                similarity = util.pytorch_cos_sim(query_embedding, content_embedding).item()
                
                if similarity > threshold:
                    matches.append({
                        'article': article,
                        'similarity': similarity
                    })
            
            # Sort by similarity score
            return sorted(matches, key=lambda x: x['similarity'], reverse=True)
            
        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            return []

class ContentParser:
    """Parse and analyze content using LLM."""
    
    def __init__(self, model_name: str = "llama3.1"):
        """Initialize with specified LLM model."""
        self.llm = OllamaLLM(model=model_name)
        
        # Define analysis prompt template
        self.template = """
        Analyze the following content and extract relevant information based on this description: {description}

        Content:
        {content}

        Format your response as follows:
        
        Key Information:
        1. [First key point]
        2. [Second key point]
        ...

        Analysis:
        [Detailed analysis of findings]

        Summary:
        [Brief summary of key takeaways]
        """
        
    def parse_articles(
        self, 
        articles: List[Dict], 
        description: str, 
        chunk_size: int = 3000
    ) -> str:
        """
        Parse and analyze articles using LLM.
        
        Args:
            articles: List of articles with similarity scores
            description: Analysis criteria
            chunk_size: Maximum content chunk size
            
        Returns:
            Parsed and analyzed content
        """
        try:
            prompt = ChatPromptTemplate.from_template(self.template)
            results = []
            
            # Process articles in chunks
            current_chunk = []
            current_size = 0
            
            for article_data in articles:
                article = article_data['article']
                article_text = f"Title: {article['title']}\nContent: {article['body']}\n"
                
                # Check if adding article would exceed chunk size
                if current_size + len(article_text) > chunk_size:
                    # Process current chunk
                    chunk_result = self._process_chunk(current_chunk, description, prompt)
                    if chunk_result:
                        results.append(chunk_result)
                    current_chunk = [article_text]
                    current_size = len(article_text)
                else:
                    current_chunk.append(article_text)
                    current_size += len(article_text)
            
            # Process final chunk
            if current_chunk:
                chunk_result = self._process_chunk(current_chunk, description, prompt)
                if chunk_result:
                    results.append(chunk_result)
            
            return "\n\n---\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Content parsing error: {str(e)}")
            return "Error occurred during content analysis."
    
    def _process_chunk(self, chunk: List[str], description: str, prompt) -> str:
        """Process a single content chunk."""
        content = "\n\n".join(chunk)
        return self.llm.invoke(prompt.format(
            description=description,
            content=content
        )).strip()
