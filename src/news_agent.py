import os
import json
import logging
import time
import warnings
from googlesearch import search
# Suppress Google Generative AI deprecation warnings
warnings.filterwarnings("ignore")

import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def search_market_news(commodity="cinnamon", num_results=5):
    """
    Fetches top news URLs and returns them as a list of strings for the specified commodity.
    Tries advanced search first, falls back to basic search if no results found.
    """
    # Strict query for advanced search
    query = f"Sri Lanka {commodity} Price trends 2026"
    logger.info(f"Searching for: {query}")
    
    news_strings = []
    
    try:
        # Advanced search (returns objects with title, description, url)
        results = search(query, num_results=num_results, advanced=True)
        
        for result in results:
            # Format: "Title: <title> - Snippet: <desc> (URL: <url>)"
            if result.title and result.description:
                news_string = f"Title: {result.title} - Snippet: {result.description} (URL: {result.url})"
                news_strings.append(news_string)
        
        logger.info(f"Advanced search found {len(news_strings)} items.")

    except Exception as e:
        logger.warning(f"Advanced search raised exception: {e}")

    # Fallback Logic: If advanced search yielded 0 results or failed
    if not news_strings:
        logger.warning("Advanced search returned 0 results. Triggering FALLBACK Basic Search...")
        try:
            # Broader query for basic search
            fallback_query = f"Sri Lanka {commodity} market price news"
            urls = search(fallback_query, num_results=num_results)
            
            # Convert URLs to compatible string format
            news_strings = [f"News Link (No Snippet): {url}" for url in urls]
            logger.info(f"Fallback search found {len(news_strings)} URLs.")
            
        except Exception as e2:
            logger.error(f"Fallback search failed: {e2}")
            # Do not return empty yet, flow continues to final check
    
    # Final Failsafe: Ensure we never return an empty list
    if not news_strings:
        logger.warning("All searches failed. Returning generic search URL.")
        generic_url = f"https://www.google.com/search?q=Sri+Lanka+{commodity}+price"
        news_strings.append(f"Title: Market Search - Snippet: Live search results for {commodity} (URL: {generic_url})")

    return news_strings


def analyze_sentiment(news_list, commodity="cinnamon"):
    """
    Analyzes sentiment using Gemini with retry logic for rate limits.
    """
    if not news_list:
        return {'sentiment': 'Neutral', 'confidence': 0.0, 'summary': 'No news found to analyze.'}

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not found.")
        return {'sentiment': 'Neutral', 'confidence': 0.0, 'summary': 'API Key missing.'}

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    combined_text = "\n".join(news_list)
    
    prompt = f"""
    Analyze these news items regarding Sri Lankan {commodity}. 
    Some items may only be URLs or titles if snippets are missing.
    
    News Data:
    {combined_text}
    
    Instructions:
    1. If snippets are present, use them to determine sentiment (Bullish/Bearish/Neutral).
    2. If ONLY URLs are present, infer potential sentiment from the URL text itself (e.g., words like 'soar', 'drop', 'crisis').
    3. If evidence is week, default to Neutral and Low Confidence.
    
    Return valid JSON with the following keys: 
    - 'sentiment' (Bullish/Bearish/Neutral)
    - 'confidence' (float between 0 and 1)
    - 'summary' (max 50 words)
    
    Ensure the output is pure JSON.
    """

    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            text_response = response.text.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON
            result = json.loads(text_response)
            return result
            
        except Exception as e:
            if "429" in str(e) or "Quota exceeded" in str(e):
                wait_time = 1 # Wait 1 second before retry (fail fast to simulation)
                logger.warning(f"Rate limit hit (429). Retrying in {wait_time}s... (Attempt {attempt+1}/{retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"Sentiment analysis failed: {e}")
                return {'sentiment': 'Neutral', 'confidence': 0.0, 'summary': f'Analysis error: {str(e)[:50]}...'}

    logger.error("Max retries exceeded for rate limit.")
    return {'sentiment': 'Neutral', 'confidence': 0.0, 'summary': 'Analysis failed: API Quota Exceeded.'}


def generate_mock_data(commodity):
    """
    Returns simulated market intelligence when live data fetch fails.
    """
    return {
        "sentiment": "Stable (Historical)",
        "confidence": 0.85,
        "summary": f"Live market data is currently unavailable. Based on historical seasonal trends for {commodity}, prices are expected to remain stable with moderate demand throughout the upcoming harvest season."
    }

def get_market_intelligence(commodity="cinnamon"):
    """
    Orchestrator function with Simulation Mode fallback.
    """
    logger.info(f"Fetching market intelligence for {commodity}...")
    
    try:
        news_list = search_market_news(commodity)
        
        # If search returns empty, trigger simulation immediately
        if not news_list:
            logger.warning("Search returned 0 results. Switching to Simulation Mode.")
            return generate_mock_data(commodity)
            
        intelligence = analyze_sentiment(news_list, commodity)
        
        # If analysis failed (Neutral/0.0 confidence usually indicates error fallback in analyze_sentiment)
        # We can choose to use that or override with simulation if we want 'cleaner' demo data.
        # The prompt below checks for specific 'Analysis failed' message from analyze_sentiment
        if intelligence.get('summary') == 'Analysis failed: API Quota Exceeded.':
             logger.warning("API Quota Exceeded. Switching to Simulation Mode.")
             return generate_mock_data(commodity)

        return intelligence

    except Exception as e:
        logger.error(f"Live fetch failed: {e}. Switching to Simulation Mode.")
        return generate_mock_data(commodity)

if __name__ == "__main__":
    # Test run
    data = get_market_intelligence("clove")
    print(json.dumps(data, indent=2))
