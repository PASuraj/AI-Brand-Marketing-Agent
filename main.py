import os
import time  
import streamlit as st
from scrape import scrape_website  
from parse import process_articles, generate_bert_embeddings, semantic_search, store_embeddings
from fireworks.client import Fireworks 
from sentence_transformers import SentenceTransformer, util
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch
import numpy as np

# Initialize AI Models
client = Fireworks(api_key="fw_xxxxxxxxxxxxxxxxxxxxx")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # For semantic similarity
sentiment_analyzer = SentimentIntensityAnalyzer()  # For sentiment & sarcasm analysis

# Function to evaluate consistency
def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to get sentence-level embeddings instead of just [CLS] token."""
    token_embeddings = model_output[0]  # Extract hidden states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def evaluate_consistency(generated_lines, reference_texts):
    """Compute deep semantic similarity between generated content and reference context."""

    # Define device before using it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Combine texts and encode using SentenceTransformer (correct method)
    all_texts = generated_lines + reference_texts
    embeddings = sbert_model.encode(all_texts, convert_to_tensor=True, device=device)

    # Normalize embeddings for cosine similarity
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Split embeddings into generated and reference sets
    gen_embeddings = embeddings[:len(generated_lines)]
    ref_embeddings = embeddings[len(generated_lines):]

    # Compute Cosine Similarity
    cosine_sim = util.pytorch_cos_sim(gen_embeddings, ref_embeddings).cpu().numpy()

    # Compute Euclidean Distance (lower distance = higher similarity)
    euclidean_dist = np.linalg.norm(gen_embeddings.cpu().numpy()[:, np.newaxis] - ref_embeddings.cpu().numpy(), axis=2)
    euclidean_sim = 1 / (1 + euclidean_dist)  # Convert distance to similarity score

    # Weighted Score (80% cosine + 20% Euclidean similarity for better balance)
    final_similarity_score = (0.8 * cosine_sim.mean()) + (0.2 * euclidean_sim.mean())

    return final_similarity_score



# Function to evaluate tone (sarcasm detection)
# Load a sarcasm detection model
sarcasm_detector = pipeline("text-classification", model="mrm8488/t5-base-finetuned-sarcasm-twitter")

def evaluate_tone(generated_lines):
    """Analyze sarcasm in generated marketing content using a fine-tuned transformer model."""
    sarcasm_scores = []
    
    for line in generated_lines:
        result = sarcasm_detector(line)
        label = result[0]['label'].lower()  # Either "sarcasm" or "not_sarcasm"
        score = result[0]['score']  # Confidence score

        # Assign higher scores for sarcastic outputs
        sarcasm_scores.append(score if "sarcasm" in label else (1 - score))

    return sum(sarcasm_scores) / len(sarcasm_scores) if sarcasm_scores else 0  # Average sarcasm score

# Function to query Fireworks for engagement rating
def evaluate_engagement(generated_lines):
    """Use Fireworks API to rate engagement level of sarcastic marketing content."""
    query = f"""
    Given the following sarcastic marketing lines, rate their audience engagement potential on a scale of 1 to 10.
    
    {generated_lines[0]}
    {generated_lines[1]}

    Respond with only a single numeric score.
    """
    
    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        messages=[{"role": "user", "content": query}]
    )
    
    time.sleep(5)  # Rate-limit API calls
    
    if hasattr(response, 'choices') and response.choices:
        try:
            return float(response.choices[0].message.content.strip())  # Convert response to float
        except ValueError:
            return 0.0  # Return 0.0 if parsing fails
    return 0.0  # Default fallback value


# Function to generate marketing content using Fireworks
def get_fireworks_response(article, similar_articles):
    """Query Fireworks for sarcastic marketing content."""
    query = f"""
    ***Your Role***:
        You are Ryan Reynolds's Deadpool—the fourth-wall-breaking, wisecracking anti-hero with a love for sarcasm and self-aware humor.
        Your job? Skewer the tech industry's overblown promises while making Niti.AI look like the savior of digital sanity. Channel Deadpool's irreverent tone, breaking the fourth wall occasionally, and use his signature blend of snarky observations with pop culture references.
    
    ***Context about Niti.AI***:
        Modern software teams continue to struggle going from idea to growth for their products.
        Our AI-first highly opinionated product development platform streamlines and accelerates execution for all stakeholders in the team, so they can bring innovation faster to their customers.
        We accelerate the digital experiences creation process by embedding AI agents and copilots throughout the software development lifecycle to educate or automate the work of every stakeholder involved in ideation, prototyping, development, and scale-up.
        We allow developers to leverage our ready-to-go SaaS templates combined with their custom data and create SaaS tooling for their business teams.
        We empower non-programmers in business or operations teams to go from innovative ideas to fully managed products & experiences in their digital products with relying on developer expertise only when absolutely needed.
        We unlock hyper-personalization for companies by allowing them to build for small segments and target delivery of digital experiences so modern AI applications become contextual and relevant for their customers.
        
    ***Your Mission (and no, you don't get to say no)***:
        Write EXACTLY two witty marketing lines that:
        Cleverly tie back to Niti.AI in a way that feels natural and unexpected - like Deadpool would make an offhand comment that suddenly connects perfectly to the product.
        Use sharp, intelligent humor to make a memorable point - aim for that perfect blend of sarcasm and insight that makes people both laugh and think.
        Stay within Niti.AI's core context—no exaggerations or made-up features. Stick to what they actually do, but present it with maximum Deadpool flair.
        You will ALWAYS FOLLOW the structure as mentioned in the ***Output Format*** section. ADD NOTHING MORE.
        
    ***Note***:
        Mention real people—pretend they're fictional characters. Roast the event, claim, or trend, not the person. Make it sound like roasting a third person.
        ALWAYS Keep it relevant to what the company actually does. Don't go off on tangents or add capabilities Niti.AI doesn't have.
        Write generic or unrelated jokes. EVERY joke must connect to both the article and Niti.AI. The humor should feel tailored, not copy-pasted.
        
    ***How to Structure Your Output***:
        ✅ ALWAYS Two bullet points. Each must be EXACTLY two lines long—short, sharp, and hilarious.
        ✅ No headings or extra text. Just the two lines of marketing content. Don't introduce them, explain them, or add any commentary.
        ✅ Balance humor with intelligence. Think extremely witty and clever, not mean-spirited. Deadpool with a marketing degree, not just Deadpool with a katana.
        
    ***Tone & Style***:
        ⚡ Self-aware humor—acknowledge how ridiculous tech hype can be. Break the fourth wall like only Deadpool can.
        ⚡ Sarcasm with a purpose—expose nonsense while making Niti.AI look ahead of the curve. The contrast should be stark and funny.
        ⚡ Keep it PG-13—funny, but not offensive. Edgy enough to feel authentic to the character but safe for all marketing contexts.
        ⚡ Use Deadpool's conversational style - casual, irreverent, with sudden shifts in tone and occasional pop culture references.
        
        Maximum effort! 🚀

    ***Article Context***:
    Title: {article['title']}
    Content Excerpt: {article['body'][:1000]}

    ***Output Format:***
    1. <First marketing line>
    
    2. <Second marketing line>
    """

    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        messages=[{"role": "user", "content": query}]
    )

    time.sleep(10)  # Rate-limit API calls

    if hasattr(response, 'choices') and response.choices:
        raw_response = response.choices[0].message.content.strip()

        # Ensure we have at least two valid lines
        marketing_lines = [line.strip() for line in raw_response.split("\n") if line.strip()]
        if len(marketing_lines) < 2:
            return raw_response, ["⚠️ Insufficient response received.", "⚠️ Please check Fireworks API output."]
        
        return raw_response, marketing_lines[:2]  # Only return first two lines
    else:
        return "No response found or unexpected response structure.", ["⚠️ No response from API.", "⚠️ Try again."]

# Streamlit UI
def main():
    st.title("AI Marketing Content Generator")

    if "articles" not in st.session_state:
        st.session_state.articles = []

    website_url = st.text_input("🔗 Enter the website URL to scrape:", "")

    if st.button("🔍 Scrape Website"):
        if website_url:
            with st.spinner("Scraping articles... Please wait."):
                articles = scrape_website(website_url)
                if articles:
                    st.session_state.articles = articles
                    st.success(f"✅ Scraped {len(articles)} articles successfully!")
                    
                    for article in articles:
                        embedding = generate_bert_embeddings(article['body'])
                        store_embeddings([article])  # Store embeddings in FAISS
                else:
                    st.warning("⚠️ No articles found. Check the URL or website structure.")

    if st.session_state.articles:
        st.write("### 📑 Scraped Articles")

        for i, article in enumerate(st.session_state.articles):
            with st.expander(f"**{i + 1}. {article['title']}**", expanded=False):  # Expandable dropdown
                st.write(f"**Full Content:**")
                st.write(article['body'])  # Displays the full article content


    if st.session_state.articles:
        if st.button("Generate Sarcastic Marketing Content"):
            st.write("### AI-Generated Marketing Content")

            for article in st.session_state.articles:
                with st.spinner(f"Generating content for: {article['title']}..."):
                    # Search FAISS for similar past articles
                    embedding = generate_bert_embeddings(article['body'])
                    similar_articles = semantic_search(article['body'])

                    # Generate content using Fireworks
                    response, generated_lines = get_fireworks_response(article, similar_articles)

                    # Reference texts (original + similar articles)
                    reference_texts = [article['body']] + [sim['body'] for sim in similar_articles]

                    # Compute Evaluation Metrics
                    consistency_score = evaluate_consistency(generated_lines, reference_texts)
                    sentiment_score = evaluate_tone(generated_lines)
                    engagement_score = evaluate_engagement(generated_lines)

                    # Display results
                    st.write(f"#### {article['title']}")
                    st.write(response)
                    st.write(f"**Consistency Score:** {consistency_score:.2f}")
                    st.write(f"**Sentiment Score (Sarcasm Detection):** {sentiment_score:.2f}")
                    st.write(f"**Engagement Score:** {engagement_score:.2f}")
                    st.write("---")

    if st.button("🔄 Refresh"):
        st.rerun()

if __name__ == "__main__":
    main()
