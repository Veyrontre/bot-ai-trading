# news.py
import requests

class NewsLoader:
    def __init__(self, api_key=None):
        # Hardcode API key langsung di sini
        self.api_key = api_key or "3618578200d04a489b84a8bd12c00779"
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch(self, query="gold OR XAUUSD OR forex", language="en", page_size=10):
        if not self.api_key:
            print("⚠️ NEWS_API_KEY not set.")
            return []
        try:
            params = {
                'apiKey': self.api_key,
                'q': query,
                'language': language,
                'pageSize': page_size,
                'sortBy': 'publishedAt'
            }
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            articles = data.get('articles', [])
            formatted_articles = []
            for article in articles:
                formatted_articles.append({
                    'publishedAt': article.get('publishedAt'),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', '')
                })
            return formatted_articles
        except Exception as e:
            print(f"[Warning] Failed to fetch news: {e}")
            return []

    def fetch_normalized_news(self, query="gold OR XAUUSD OR forex", page_size=5):
        raw = self.fetch(query=query, page_size=page_size)
        normalized = []
        for a in raw:
            normalized.append({
                'timestamp': a.get('publishedAt'),
                'source': a.get('source'),
                'headline': a.get('title'),
                'summary': a.get('description'),
                'content': a.get('content'),
                'sentiment': 'neutral'
            })
        return normalized
