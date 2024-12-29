import streamlit as st
from typing import List, Dict
import logging
from scrape import scrape_website
from parse import SemanticProcessor, ContentParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraperApp:
    """Main application class for web scraping and analysis."""

    def __init__(self):
        """Initialize application components."""
        self.semantic_processor = SemanticProcessor()
        self.content_parser = ContentParser()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Set up initial session state variables."""
        if "articles" not in st.session_state:
            st.session_state.articles = []
        if "analysis_description" not in st.session_state:
            st.session_state.analysis_description = ""

    def run(self):
        """Run the Streamlit application."""
        st.title("AI Web Content Analyzer")

        # Scraping Section
        with st.container():
            st.subheader("Web Scraping")
            url = st.text_input("Enter Website URL")

            if st.button("Scrape Content"):
                if url:
                    with st.spinner("Scraping website..."):
                        try:
                            articles = scrape_website(url)
                            st.session_state.articles = articles
                            st.success(f"Successfully scraped {len(articles)} articles!")
                        except Exception as e:
                            st.error(f"Error scraping website: {str(e)}")

        # Display scraped content
        if st.session_state.articles:
            with st.expander("View Scraped Content"):
                for idx, article in enumerate(st.session_state.articles, 1):
                    st.markdown(f"**Article {idx}**")
                    st.write(f"Title: {article['title']}")
                    st.write(f"Preview: {article['body'][:200]}...")
                    st.markdown("---")

        # Analysis Section
        if st.session_state.articles:
            with st.container():
                st.subheader("2. Content Analysis")

                # Analysis parameters
                description = st.text_area(
                    "What information are you looking for?",
                    st.session_state.analysis_description
                )

                col1, col2 = st.columns(2)
                with col1:
                    threshold = st.slider(
                        "Similarity Threshold",
                        0.1, 1.0, 0.5, 0.1,
                        help="Higher values mean stricter matching"
                    )
                with col2:
                    top_k = st.number_input(
                        "Maximum Results",
                        min_value=1,
                        max_value=20,
                        value=5,
                        help="Maximum number of articles to analyze"
                    )

                if st.button("Analyze Content"):
                    if description:
                        self.analyze_content(description, threshold, top_k)
                    else:
                        st.warning("Please describe what information you're looking for.")

    def analyze_content(self, description: str, threshold: float, top_k: int):
        """
        Process content analysis request.

        Args:
            description: Analysis criteria
            threshold: Similarity threshold
            top_k: Maximum number of results
        """
        try:
            with st.spinner("Analyzing content..."):
                # Semantic search
                matches = self.semantic_processor.search_articles(
                    st.session_state.articles,
                    description,
                    threshold
                )[:top_k]

                if matches:
                    st.write(f"Found {len(matches)} relevant articles")

                    # Parse and analyze content
                    analysis = self.content_parser.parse_articles(
                        matches,
                        description
                    )

                    # Display results
                    st.subheader("Analysis Results")
                    st.write(analysis)

                    # Show matching articles
                    with st.expander("View Matching Articles"):
                        for idx, match in enumerate(matches, 1):
                            st.markdown(f"**Match {idx}** (Similarity: {match['similarity']:.2f})")
                            st.write(f"Title: {match['article']['title']}")
                            st.write(f"Content: {match['article']['body'][:200]}...")
                            st.markdown("---")
                else:
                    st.warning("No matching content found. Try adjusting the similarity threshold.")

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            st.error("Error occurred during analysis.")

if __name__ == "__main__":
    app = WebScraperApp()
    app.run()
