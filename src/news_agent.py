import os
import json
import logging
from googlesearch import search
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def search_market_news(query="Sri Lanka Cinnamon Price trends 2026", num_results=5):
    """
    Fetches top news URLs and returns them as a list of strings.
    In a real scenario, we might use a snippet extractor. 
    Here we return 'Title - URL' as the string representation.
    """
    logger.info(f"Searching for: {query}")
    try:
        # standard search returns URLs
        results = search(query, num_results=num_results, advanced=True)
        news_strings = []
        for result in results:
            # Format: "Title: <title> - Snippet: <desc> (URL: <url>)"
            news_string = f"Title: {result.title} - Snippet: {result.description} (URL: {result.url})"
            news_strings.append(news_string)
            
        logger.info(f"Found {len(news_strings)} news items.")
        return news_strings
    except Exception as e:
        logger.warning(f"Advanced search failed: {e}. Falling back to basic search.")
        try:
            urls = search(query, num_results=num_results)
            return [f"News Link: {url}" for url in urls]
        except Exception as e2:
            logger.error(f"Search failed completely: {e2}")
            return []

def analyze_sentiment(news_list):
    """
    Analyzes sentiment using Gemini.
    """
    if not news_list:
        return {'sentiment': 'Neutral', 'confidence': 0.0, 'summary': 'No news found to analyze.'}

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not found.")
        return {'sentiment': 'Neutral', 'confidence': 0.0, 'summary': 'API Key missing.'}

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        combined_text = "\n".join(news_list)
        
        prompt = f"""
        Analyze these news snippets regarding Sri Lankan cinnamon. 
        
        News Snippets:
        {combined_text}
        
        Return valid JSON with the following keys: 
        - 'sentiment' (Bullish/Bearish/Neutral)
        - 'confidence' (float between 0 and 1)
        - 'summary' (max 50 words)
        
        Ensure the output is pure JSON.
        """
        
        response = model.generate_content(prompt)
        text_response = response.text.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON
        result = json.loads(text_response)
        return result
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {'sentiment': 'Neutral', 'confidence': 0.0, 'summary': 'Analysis failed due to an error.'}

def get_market_intelligence():
    """
    Orchestrator function.
    """
    logger.info("Fetching market intelligence...")
    news_list = search_market_news()
    intelligence = analyze_sentiment(news_list)
    return intelligence

if __name__ == "__main__":
    # Test run
    data = get_market_intelligence()
    print(json.dumps(data, indent=2))
