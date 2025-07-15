# 🌟 Key Features
# Data Sources:

* ArXiv.org: Scrapes latest ML/AI research papers from categories like cs.AI, cs.LG, cs.CV

* TechCrunch: Fetches AI-related news articles from their AI section

* Analytics Vidhya: Scrapes latest ML tutorials and articles

# Core Functionality:

Real-time scraping with intelligent caching (1-hour cache to avoid rate limits)
Concurrent fetching for faster data retrieval
Trending keywords analysis using NLTK for text processing
Interactive visualizations with Plotly for keyword trends and source distribution
Auto-refresh capability with customizable intervals
Responsive design with custom CSS styling.

# User Interface:

Clean, modern design with gradient headers and card layouts
Sidebar controls for filtering sources and settings
Metrics dashboard showing article counts
Grouped article display by source
Search and filter capabilities

# 🚀 Installation & Setup
To run this project, you'll need to install the required dependencies:
bashpip install streamlit requests beautifulsoup4 pandas plotly nltk feedparser

📁 Project Structure

ai-trends-monitor/

├── app.py                  # Main Streamlit application

├── requirements.txt       # Dependencies

└── README.md             # Project documentation
💡 Usage

* Run the application:

        streamlit run app.py

# Features available:

* Toggle auto-refresh for real-time updates
* Filter by specific sources
* Adjust number of articles per source
* View trending keywords visualization
* Click article titles to read full content



# 🔧 Technical Highlights

Robust error handling for network issues and parsing errors
Rate limiting and caching to respect website policies
Concurrent processing for faster data fetching
Text analysis for keyword extraction and trend identification
Responsive layout that works on different screen sizes

# Live Demo 
    https://ai-tech-trend-web-scraper-eef2f6h5jgzutudnv2otzo.streamlit.app/