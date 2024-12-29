  import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from urllib.parse import urlparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        """Extract all article links from the given page and follow them."""
        article_links = response.css('a::attr(href)').getall() or []
        article_links = [response.urljoin(link) for link in article_links if link]

        # Restrict links to the same domain
        article_links = [link for link in article_links if urlparse(link).netloc == self.allowed_domain]

        logger.info(f"Found {len(article_links)} links on the page: {response.url}")

        # Follow each article link (no recursion)
        for link in article_links:
            yield scrapy.Request(link, callback=self.parse_article)

    def parse_article(self, response):
        """Extract article title and body."""
        title = response.css('h1::text').get() or response.css('title::text').get() or "No title found"
        paragraphs = response.css('article p::text, .content p::text, .post p::text').getall()
        body = ' '.join(p.strip() for p in paragraphs if p.strip()) if paragraphs else "No content found"

        if title.strip() and body.strip():
            logger.info(f"Scraped article: {title[:50]}...")
            self.articles.append({'title': title.strip(), 'body': body.strip()})
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

    # Shared list to collect the articles
    articles = []

    # Set up the process and configure Scrapy settings
    process = CrawlerProcess(get_project_settings())  # Use Scrapy's project settings
    process.crawl(ArticleSpider, start_url=url, articles_list=articles)  # Pass start_url and articles list
    process.start()  # Block here until the crawling finishes

    return articles
