import os
import json
import datetime
from textblob import TextBlob
from googlesearch import search
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
QUERY = "Sri Lanka Cinnamon Price trends 2026"
OUTPUT_FILE = "data/processed/latest_sentiment.json"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def fetch_market_news(query=QUERY, num_results=5):
    """
    Fetches top news URLs using Google Search.
    Note: googlesearch-python returns iterators of URLs. 
    Advanced search with snippets requires different handling or libraries.
    We will stick to URLs and simulate snippet extraction or use description if available in specific search functions.
    Standard 'search' yields URLs.
    """
    logger.info(f"Searching for: {query}")
    try:
        # advanced=True yields SearchResult objects with .title, .description, .url
        # If the installed library is 'googlesearch-python', it supports advanced=True
        results = search(query, num_results=num_results, advanced=True)
        news_items = []
        for result in results:
            news_items.append({
                'title': result.title,
                'description': result.description,
                'url': result.url
            })
        logger.info(f"Found {len(news_items)} news items.")
        return news_items
    except Exception as e:
        logger.warning(f"Advanced search failed or not supported: {e}. Falling back to URL-only search.")
        try:
            urls = search(query, num_results=num_results)
            return [{'url': url, 'title': 'Unknown Title', 'description': ''} for url in urls]
        except Exception as e2:
            logger.error(f"Search failed completely: {e2}")
            return []

def analyze_sentiment(news_items):
    """
    Analyzes sentiment using Gemini (primary) or TextBlob (fallback).
    """
    if not news_items:
        return {'sentiment_score': 0, 'summary': "No news found.", 'source_urls': []}

    # Prepare text for analysis
    headlines = [f"- {item.get('title', '')}: {item.get('description', '')} ({item.get('url', '')})" for item in news_items]
    combined_text = "\n".join(headlines)
    source_urls = [item.get('url') for item in news_items]

    # Primary: Gemini
    if GEMINI_API_KEY:
        try:
            logger.info("Using Gemini for sentiment analysis...")
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            Analyze the following market news headlines regarding Sri Lanka Cinnamon Price trends for 2026:
            
            {combined_text}
            
            Return a JSON object with the following fields:
            - 'sentiment_score': A float between -1 (Very Bearish) and 1 (Very Bullish).
            - 'summary': A concise summary of the market outlook based on these headlines.
            
            JSON:
            """
            
            response = model.generate_content(prompt)
            # Cleanup json string if markdown code blocks are present
            text_response = response.text.replace('```json', '').replace('```', '').strip()
            result = json.loads(text_response)
            
            # Ensure fields exist
            result['source_urls'] = source_urls
            return result
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}. Switching to fallback.")
    
    # Fallback: TextBlob
    logger.info("Using TextBlob for sentiment analysis (fallback)...")
    scores = []
    for item in news_items:
        text = f"{item.get('title', '')} {item.get('description', '')}"
        blob = TextBlob(text)
        scores.append(blob.sentiment.polarity)
    
    avg_score = sum(scores) / len(scores) if scores else 0
    
    summary = "Market sentiment appears "
    if avg_score > 0.1:
        summary += "bullish based on recent headlines."
    elif avg_score < -0.1:
        summary += "bearish based on recent headlines."
    else:
        summary += "neutral/mixed based on recent headlines."
        
    return {
        'sentiment_score': avg_score,
        'summary': summary + " (Generated via TextBlob Fallback)",
        'source_urls': source_urls
    }

def save_intelligence(data):
    """
    Saves the sentiment intelligence to a JSON file.
    """
    output_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'sentiment_score': data.get('sentiment_score', 0),
        'summary': data.get('summary', 'No summary available'),
        'source_urls': data.get('source_urls', [])
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=4)
    logger.info(f"Intelligence saved to {OUTPUT_FILE}")

def main():
    logger.info("Starting Market Intelligence Agent...")
    news_items = fetch_market_news()
    sentiment_data = analyze_sentiment(news_items)
    save_intelligence(sentiment_data)
    logger.info("Agent run completed.")

if __name__ == "__main__":
    main()
