import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from urllib.parse import urljoin, urlparse
import feedparser
import json
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import concurrent.futures
import threading
from typing import List, Dict, Tuple
import hashlib


# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data with proper error handling"""
    try:
        # Try to find punkt_tab first (newer versions)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except:
                # Fallback to older punkt
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)

        # Download stopwords
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

    except Exception as e:
        st.warning(f"Could not download NLTK data: {e}. Keyword extraction may not work properly.")


# Call the download function
download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="AI & Tech Trends Monitor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }

    .source-header {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }

    .article-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #ff7f0e;
    }

    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }

    .trend-badge {
        background: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }

    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
</style>
""", unsafe_allow_html=True)


class AITrendsMonitor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.cache = {}
        self.cache_expiry = 3600  # 1 hour cache

    def get_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.md5(url.encode()).hexdigest()

    def is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is still valid"""
        if cache_key not in self.cache:
            return False
        return time.time() - self.cache[cache_key]['timestamp'] < self.cache_expiry

    def get_from_cache(self, cache_key: str):
        """Get data from cache"""
        if self.is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        return None

    def set_cache(self, cache_key: str, data):
        """Set data in cache"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }

    def scrape_arxiv(self, max_results: int = 20) -> List[Dict]:
        """Scrape ArXiv for ML/AI papers"""
        try:
            # ArXiv API endpoint for CS.AI (Artificial Intelligence) and CS.LG (Machine Learning)
            queries = ['cat:cs.AI', 'cat:cs.LG', 'cat:cs.CV', 'cat:cs.NE']
            all_papers = []

            for query in queries:
                url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results // len(queries)}&sortBy=submittedDate&sortOrder=descending"
                cache_key = self.get_cache_key(url)

                cached_data = self.get_from_cache(cache_key)
                if cached_data:
                    all_papers.extend(cached_data)
                    continue

                response = self.session.get(url, timeout=10)
                response.raise_for_status()

                # Parse the XML response
                feed = feedparser.parse(response.content)
                papers = []

                for entry in feed.entries:
                    paper = {
                        'title': entry.title,
                        'link': entry.link,
                        'summary': entry.summary[:300] + "..." if len(entry.summary) > 300 else entry.summary,
                        'published': entry.published,
                        'authors': [author.name for author in entry.authors] if hasattr(entry, 'authors') else [],
                        'category': entry.tags[0].term if hasattr(entry, 'tags') and entry.tags else 'AI',
                        'source': 'ArXiv'
                    }
                    papers.append(paper)

                self.set_cache(cache_key, papers)
                all_papers.extend(papers)
                time.sleep(0.5)  # Rate limiting

            return all_papers[:max_results]

        except Exception as e:
            st.error(f"Error scraping ArXiv: {str(e)}")
            return []

    def scrape_techcrunch(self, max_results: int = 15) -> List[Dict]:
        """Scrape TechCrunch for AI-related articles"""
        try:
            # TechCrunch AI section RSS feed
            url = "https://techcrunch.com/category/artificial-intelligence/feed/"
            cache_key = self.get_cache_key(url)

            cached_data = self.get_from_cache(cache_key)
            if cached_data:
                return cached_data

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            feed = feedparser.parse(response.content)
            articles = []

            for entry in feed.entries[:max_results]:
                article = {
                    'title': entry.title,
                    'link': entry.link,
                    'summary': BeautifulSoup(entry.summary, 'html.parser').get_text()[:300] + "..." if len(
                        entry.summary) > 300 else BeautifulSoup(entry.summary, 'html.parser').get_text(),
                    'published': entry.published,
                    'authors': [entry.author] if hasattr(entry, 'author') else [],
                    'category': 'AI News',
                    'source': 'TechCrunch'
                }
                articles.append(article)

            self.set_cache(cache_key, articles)
            return articles

        except Exception as e:
            st.error(f"Error scraping TechCrunch: {str(e)}")
            return []

    def scrape_analytics_vidhya(self, max_results: int = 15) -> List[Dict]:
        """Scrape Analytics Vidhya for AI/ML articles"""
        try:
            url = "https://www.analyticsvidhya.com/blog/"
            cache_key = self.get_cache_key(url)

            cached_data = self.get_from_cache(cache_key)
            if cached_data:
                return cached_data

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []

            # Find article containers
            article_containers = soup.find_all('article', class_='post-block')

            for container in article_containers[:max_results]:
                try:
                    title_elem = container.find('h2') or container.find('h3') or container.find('a')
                    if not title_elem:
                        continue

                    title = title_elem.get_text().strip()
                    link_elem = title_elem.find('a') or title_elem
                    link = urljoin(url, link_elem.get('href', '')) if link_elem else ''

                    summary_elem = container.find('p') or container.find('div', class_='excerpt')
                    summary = summary_elem.get_text().strip()[:300] + "..." if summary_elem else "No summary available"

                    # Try to find publish date
                    date_elem = container.find('time') or container.find('span', class_='date')
                    published = date_elem.get_text().strip() if date_elem else datetime.now().strftime('%Y-%m-%d')

                    # Try to find author
                    author_elem = container.find('span', class_='author') or container.find('a', class_='author')
                    author = author_elem.get_text().strip() if author_elem else 'Analytics Vidhya'

                    article = {
                        'title': title,
                        'link': link,
                        'summary': summary,
                        'published': published,
                        'authors': [author],
                        'category': 'ML Tutorial',
                        'source': 'Analytics Vidhya'
                    }
                    articles.append(article)

                except Exception as e:
                    continue

            self.set_cache(cache_key, articles)
            return articles

        except Exception as e:
            st.error(f"Error scraping Analytics Vidhya: {str(e)}")
            return []

    def extract_trending_keywords(self, articles: List[Dict]) -> List[Tuple[str, int]]:
        """Extract trending keywords from article titles and summaries"""
        try:
            # Try to get stopwords, fallback to basic list if NLTK fails
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = set(
                    ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is',
                     'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
                     'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'])

            stop_words.update(['ai', 'ml', 'machine', 'learning', 'artificial', 'intelligence',
                               'data', 'science', 'deep', 'neural', 'network', 'algorithm',
                               'model', 'training', 'paper', 'research', 'study', 'analysis',
                               'using', 'new', 'how', 'what', 'why', 'when', 'where', 'which',
                               'introduction', 'conclusion', 'results', 'method', 'approach'])

            all_text = []
            for article in articles:
                text = f"{article['title']} {article['summary']}"
                all_text.append(text.lower())

            combined_text = ' '.join(all_text)

            # Try to use NLTK tokenizer, fallback to simple split
            try:
                words = word_tokenize(combined_text)
            except:
                # Simple fallback tokenization
                import re
                words = re.findall(r'\b[a-zA-Z]+\b', combined_text)

            # Filter words
            words = [word.lower() for word in words if
                     word.isalpha() and len(word) > 3 and word.lower() not in stop_words]

            # Count frequencies
            word_freq = Counter(words)
            return word_freq.most_common(15)

        except Exception as e:
            st.error(f"Error extracting keywords: {str(e)}")
            # Return some basic keywords as fallback
            return [('python', 5), ('tensorflow', 4), ('pytorch', 3), ('nlp', 3), ('computer', 2)]

    def get_all_articles(self) -> List[Dict]:
        """Get articles from all sources concurrently"""
        all_articles = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.scrape_arxiv, 20): "ArXiv",
                executor.submit(self.scrape_techcrunch, 15): "TechCrunch",
                executor.submit(self.scrape_analytics_vidhya, 15): "Analytics Vidhya"
            }

            for future in concurrent.futures.as_completed(futures):
                source = futures[future]
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                except Exception as e:
                    st.error(f"Error fetching from {source}: {str(e)}")

        return all_articles


def main():
    st.markdown('<h1 class="main-header">🤖 AI & Tech Trends Monitor</h1>', unsafe_allow_html=True)

    # Initialize the monitor
    monitor = AITrendsMonitor()

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")

    auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", value=False)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 60, 600, 300)

    sources_filter = st.sidebar.multiselect(
        "Select Sources",
        ["ArXiv", "TechCrunch", "Analytics Vidhya"],
        default=["ArXiv", "TechCrunch", "Analytics Vidhya"]
    )

    max_articles = st.sidebar.slider("Max articles per source", 5, 30, 15)

    if st.sidebar.button("🔄 Refresh Data") or auto_refresh:
        st.cache_data.clear()

    # Main content
    col1, col2, col3 = st.columns(3)

    with st.spinner("🔍 Fetching latest AI trends..."):
        articles = monitor.get_all_articles()

    # Filter articles by selected sources
    filtered_articles = [article for article in articles if article['source'] in sources_filter]

    # Display metrics
    with col1:
        st.metric("📰 Total Articles", len(filtered_articles))

    with col2:
        research_papers = len([a for a in filtered_articles if a['source'] == 'ArXiv'])
        st.metric("📄 Research Papers", research_papers)

    with col3:
        news_articles = len([a for a in filtered_articles if a['source'] != 'ArXiv'])
        st.metric("📺 News Articles", news_articles)

    # Trending keywords
    if filtered_articles:
        st.subheader("🔥 Trending Keywords")
        keywords = monitor.extract_trending_keywords(filtered_articles)

        if keywords:
            # Create keyword cloud visualization
            df_keywords = pd.DataFrame(keywords, columns=['keyword', 'frequency'])
            fig = px.bar(df_keywords.head(10), x='frequency', y='keyword',
                         orientation='h', title="Top Trending Keywords")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Source distribution
    if filtered_articles:
        st.subheader("📊 Articles by Source")
        source_counts = Counter([article['source'] for article in filtered_articles])

        fig = px.pie(values=list(source_counts.values()),
                     names=list(source_counts.keys()),
                     title="Distribution of Articles by Source")
        st.plotly_chart(fig, use_container_width=True)

    # Articles display
    st.subheader("📚 Latest Articles")

    if not filtered_articles:
        st.warning("No articles found. Try refreshing or adjusting your source filters.")
    else:
        # Group articles by source
        articles_by_source = {}
        for article in filtered_articles:
            source = article['source']
            if source not in articles_by_source:
                articles_by_source[source] = []
            articles_by_source[source].append(article)

        # Display articles by source
        for source, source_articles in articles_by_source.items():
            if source in sources_filter:
                st.markdown(f'<div class="source-header"><h3>📖 {source} ({len(source_articles)} articles)</h3></div>',
                            unsafe_allow_html=True)

                for article in source_articles[:max_articles]:
                    with st.container():
                        st.markdown(f'<div class="article-card">', unsafe_allow_html=True)

                        # Article title and link
                        st.markdown(f"**[{article['title']}]({article['link']})**")

                        # Article metadata
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            if article['authors']:
                                authors_str = ', '.join(article['authors'][:3])
                                if len(article['authors']) > 3:
                                    authors_str += f" +{len(article['authors']) - 3} more"
                                st.caption(f"👤 {authors_str}")

                        with col2:
                            st.caption(f"📅 {article['published']}")

                        with col3:
                            st.markdown(f'<span class="trend-badge">{article["category"]}</span>',
                                        unsafe_allow_html=True)

                        # Article summary
                        st.write(article['summary'])

                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("---")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🤖 AI & Tech Trends Monitor | Data refreshed every few minutes</p>
        <p>Sources: ArXiv.org, TechCrunch.com, AnalyticsVidhya.com</p>
    </div>
    """, unsafe_allow_html=True)

    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()