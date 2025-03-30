import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from urllib.parse import urlparse
import logging
import re
import unicodedata
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FAISS and embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)  # Adjust dimension based on model output
vector_db = {}  # Dictionary to store text embeddings

class ArticleSpider(scrapy.Spider):
    name = "article_spider"
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 1,
    }

    def __init__(self, start_url=None, articles_list=None, *args, **kwargs):
        """Initialize the spider with a start URL and an articles list."""
        super().__init__(*args, **kwargs)
        if not start_url or not isinstance(start_url, str):
            raise ValueError("A valid 'start_url' (string) must be provided.")
        self.start_url = start_url
        self.start_urls = [start_url]
        self.allowed_domain = urlparse(start_url).netloc  # Restrict to the same domain
        self.articles = articles_list  # Shared list for storing articles

    def parse(self, response):
        """Extract article links from the page and follow them."""
        all_links = response.css('a::attr(href)').getall()
        if not all_links:
            return
        
        # Join relative URLs with the base URL
        absolute_links = [response.urljoin(link) for link in all_links]
        
        # Filter links that likely lead to articles
        article_links = [
            link for link in absolute_links
            if re.search(r'(\/article\/|\/news\/|\/story\/|\.html|\.php)', link)
            and urlparse(link).netloc == self.allowed_domain  # Restrict to the same domain
        ]
        
        logger.info(f"Found {len(article_links)} potential article links on the page: {response.url}")
        
        # Follow each article link
        for link in article_links:
            yield scrapy.Request(link, callback=self.parse_article)

    def parse_article(self, response):
        """Extract article title and body."""
        title = response.css('h1::text, title::text').get() or "No title found"

        # Extract body text while filtering out non-essential elements
        paragraphs = response.css('article p::text, section p::text, div p::text, p::text').getall()
        body = ' '.join(p.strip() for p in paragraphs if p.strip()) if paragraphs else "No content found"
        
        # Normalize unicode and remove excessive whitespace
        title = unicodedata.normalize("NFKC", title.strip())
        body = unicodedata.normalize("NFKC", body.strip())

        if title and body and len(body) > 200:  # Avoid storing very short content
            logger.info(f"Scraped article: {title[:50]}...")  # Log first 50 chars of title
            self.articles.append({'title': title, 'body': body})
        else:
            logger.warning(f"Skipping article with insufficient data: {response.url}")

def scrape_website(url: str):
    """
    Scrape articles from a website and return a list of articles.
    
    Args:
        url (str): The starting URL to scrape.
        
    Returns:
        List[Dict]: A list of articles with titles and bodies
    """
    if not url or not isinstance(url, str):
        raise ValueError("A valid URL (string) must be provided for scraping.")
    
    articles = []
    
    try:
        # Set up Scrapy crawler process
        process = CrawlerProcess(get_project_settings())
        process.crawl(ArticleSpider, start_url=url, articles_list=articles)
        process.start()  # Blocking execution until crawl is finished
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
    
    return articles
