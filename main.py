import os
import streamlit as st
from scrape import scrape_website  # Assumes this function exists to scrape articles
from parse import process_articles  # Assumes this function exists to parse articles
from openai import OpenAI  # Importing OpenAI for DeepSeek integration

# Initialize the DeepSeek client
client = OpenAI(api_key="DEEPSEEK-API-KEY", base_url="https://api.deepseek.com")

def get_deepseek_response(query, context):
    """Query the DeepSeek chat model synchronously with a prompt and context."""
    try:
        # Call DeepSeek's chat completions API
        response = client.chat.completions.create(
            model="deepseek-chat",  # Use the DeepSeek model
            messages=[
                {"role": "system", "content": "You are an assistant helping answer questions based on scraped articles."},
                {"role": "user", "content": f"Answer the following question based on the provided articles:\n\n{context}\n\nQuestion: {query}"}
            ],
            stream=False  # Set to False for a single response
        )

        # Inspect the structure of the response
        print(response)  # Check the response structure in the terminal/log

        # Attempt to access the content safely (adjust based on the actual structure)
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content.strip()
        else:
            return "No response found or unexpected response structure."

    except Exception as e:
        return f"Error occurred: {str(e)}"

def main():
    st.title("Web Scraping and DeepSeek Query")

    # Initialize session state to store scraped articles
    if "articles" not in st.session_state:
        st.session_state.articles = []

    # User input for the URL
    website_url = st.text_input("Enter the website URL to scrape:", "")

    # Scrape button
    if st.button("Scrape Website"):
        if website_url:
            st.write(f"Scraping data from: {website_url}")
            articles = scrape_website(website_url)  # Scrape articles from the URL

            if articles:
                st.session_state.articles = articles  # Store articles in session state
                st.success(f"Scraped {len(articles)} articles successfully!")
            else:
                st.warning("No articles found. Please check the URL or website structure.")

    # Display scraped articles (if any)
    if st.session_state.articles:
        st.write(f"**Total Articles Scraped:** {len(st.session_state.articles)}")

        # Collapsible display for all articles
        with st.expander("View All Scraped Articles"):
            for i, article in enumerate(st.session_state.articles):
                st.write(f"**{i + 1}. {article['title']}**")
                st.write(f"**Summary:** {article['body'][:500]}...")  # Preview first 500 characters
                st.write("---")

        # Ask for user query
        query = st.text_input("Enter your query:", "")

        # Submit button for query
        if st.button("Submit Query"):
            if query:
                st.write(f"Performing DeepSeek query: {query}")

                # Combine the content of all articles for context
                context = " ".join([article['body'] for article in st.session_state.articles[:3]])  # Use first 3 articles for context

                try:
                    # Get DeepSeek's response to the query (synchronously)
                    response = get_deepseek_response(query, context)

                    # Display the response
                    st.write("### DeepSeek Response:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error in processing query with DeepSeek: {e}")
            else:
                st.warning("Please enter a query to proceed.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
