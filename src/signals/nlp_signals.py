"""
NLP-based Signal Generation
Uses FinBERT, sentiment analysis, and news processing for trading signals.
"""

import pandas as pd
import numpy as np
import requests
import json
import re
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings
from textblob import TextBlob
import yfinance as yf
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import asyncio
import aiohttp
from dataclasses import dataclass
import os

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

@dataclass
class NewsItem:
    """News item data structure."""
    title: str
    description: str
    content: str
    published_at: datetime
    source: str
    url: str
    sentiment_score: float = 0.0
    relevance_score: float = 0.0

class FinancialNLPAnalyzer:
    """Advanced NLP analyzer for financial text."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.sentiment_analyzer = None
        self.finbert_model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self._initialize_models()
        
        # Financial keywords for relevance scoring
        self.financial_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'loss', 'quarterly', 'annual'],
            'guidance': ['guidance', 'outlook', 'forecast', 'expectations', 'projections'],
            'analyst': ['analyst', 'rating', 'upgrade', 'downgrade', 'target', 'price'],
            'market': ['market', 'trading', 'volume', 'price', 'stock', 'shares'],
            'economic': ['economy', 'inflation', 'interest', 'rates', 'fed', 'policy'],
            'sector': ['sector', 'industry', 'competition', 'market share', 'growth'],
            'risk': ['risk', 'volatility', 'uncertainty', 'concern', 'challenge'],
            'opportunity': ['opportunity', 'growth', 'expansion', 'acquisition', 'merger']
        }
        
    def _initialize_models(self):
        """Initialize NLP models."""
        try:
            # Initialize FinBERT for financial sentiment
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.finbert_model.to(self.device)
            self.finbert_model.eval()
            
            # Initialize general sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("NLP models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize advanced NLP models: {e}")
            logger.info("Falling back to TextBlob for sentiment analysis")
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of financial text."""
        if not text or len(text.strip()) < 10:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        try:
            # Try FinBERT first
            if self.finbert_model and self.tokenizer:
                return self._analyze_finbert_sentiment(text)
            
            # Fallback to general sentiment analyzer
            elif self.sentiment_analyzer:
                return self._analyze_general_sentiment(text)
            
            # Final fallback to TextBlob
            else:
                return self._analyze_textblob_sentiment(text)
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._analyze_textblob_sentiment(text)
    
    def _analyze_finbert_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using FinBERT."""
        # Truncate text if too long
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.finbert_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            
        # FinBERT labels: positive, negative, neutral
        labels = ['negative', 'neutral', 'positive']
        scores = probabilities[0].cpu().numpy()
        
        # Get the highest scoring label
        max_idx = np.argmax(scores)
        sentiment = labels[max_idx]
        confidence = float(scores[max_idx])
        
        # Convert to numerical score (-1 to 1)
        if sentiment == 'positive':
            score = confidence
        elif sentiment == 'negative':
            score = -confidence
        else:
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': confidence,
            'model': 'finbert'
        }
    
    def _analyze_general_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using general sentiment model."""
        result = self.sentiment_analyzer(text[:512])[0]
        
        label = result['label'].lower()
        confidence = result['score']
        
        # Convert labels to numerical score
        if 'positive' in label:
            score = confidence
        elif 'negative' in label:
            score = -confidence
        else:
            score = 0.0
        
        return {
            'sentiment': label,
            'score': score,
            'confidence': confidence,
            'model': 'general'
        }
    
    def _analyze_textblob_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Convert polarity to sentiment label
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': polarity,
            'confidence': 1 - subjectivity,  # Higher confidence for less subjective text
            'model': 'textblob'
        }
    
    def calculate_relevance_score(self, text: str, symbol: str = None) -> float:
        """Calculate relevance score for financial text."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        relevance_score = 0.0
        
        # Check for financial keywords
        for category, keywords in self.financial_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    relevance_score += 0.1
                    break
        
        # Check for specific symbol mentions
        if symbol:
            symbol_patterns = [symbol.lower(), symbol.lower().replace('.', ''), 
                             f"${symbol.lower()}", f"${symbol.upper()}"]
            for pattern in symbol_patterns:
                if pattern in text_lower:
                    relevance_score += 0.3
                    break
        
        # Check for numbers (prices, percentages, etc.)
        number_patterns = [
            r'\$\d+\.?\d*',  # Dollar amounts
            r'\d+\.?\d*%',   # Percentages
            r'\d+\.?\d*[mb]illion',  # Large numbers
            r'\d+\.?\d*[k]',  # Thousands
        ]
        
        for pattern in number_patterns:
            if re.search(pattern, text_lower):
                relevance_score += 0.1
                break
        
        # Normalize score
        return min(relevance_score, 1.0)
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from financial text."""
        if not text:
            return []
        
        # Tokenize and clean text
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        # Remove stop words and lemmatize
        cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens 
                         if token.isalnum() and token not in stop_words]
        
        # Find n-grams (2-3 word phrases)
        phrases = []
        for n in [2, 3]:
            for i in range(len(cleaned_tokens) - n + 1):
                phrase = ' '.join(cleaned_tokens[i:i+n])
                if len(phrase) > 3:  # Filter out very short phrases
                    phrases.append(phrase)
        
        # Count phrase frequency
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Return top phrases
        sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, count in sorted_phrases[:max_phrases]]

class NewsDataFetcher:
    """Fetches and processes financial news data."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.api_keys = {
            'newsapi': os.getenv('NEWS_API_KEY', ''),
            'alphavantage': os.getenv('ALPHA_VANTAGE_API_KEY', ''),
            'polygon': os.getenv('POLYGON_API_KEY', '')
        }
        
    async def fetch_news_async(self, symbol: str, days_back: int = 7) -> List[NewsItem]:
        """Fetch news asynchronously from multiple sources."""
        tasks = []
        
        # NewsAPI
        if self.api_keys['newsapi']:
            tasks.append(self._fetch_newsapi_news(symbol, days_back))
        
        # Alpha Vantage
        if self.api_keys['alphavantage']:
            tasks.append(self._fetch_alphavantage_news(symbol, days_back))
        
        # Polygon
        if self.api_keys['polygon']:
            tasks.append(self._fetch_polygon_news(symbol, days_back))
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine and deduplicate news items
        all_news = []
        for result in results:
            if isinstance(result, list):
                all_news.extend(result)
        
        return self._deduplicate_news(all_news)
    
    async def _fetch_newsapi_news(self, symbol: str, days_back: int) -> List[NewsItem]:
        """Fetch news from NewsAPI."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f'"{symbol}" OR "{symbol} stock" OR "{symbol} earnings"',
                    'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'apiKey': self.api_keys['newsapi']
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        news_items = []
                        
                        for article in data.get('articles', []):
                            news_item = NewsItem(
                                title=article.get('title', ''),
                                description=article.get('description', ''),
                                content=article.get('content', ''),
                                published_at=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                                source=article.get('source', {}).get('name', ''),
                                url=article.get('url', '')
                            )
                            news_items.append(news_item)
                        
                        return news_items
                    else:
                        logger.warning(f"NewsAPI request failed with status {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching NewsAPI news: {e}")
            return []
    
    async def _fetch_alphavantage_news(self, symbol: str, days_back: int) -> List[NewsItem]:
        """Fetch news from Alpha Vantage."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': symbol,
                    'apikey': self.api_keys['alphavantage']
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        news_items = []
                        
                        for article in data.get('feed', []):
                            news_item = NewsItem(
                                title=article.get('title', ''),
                                description=article.get('summary', ''),
                                content=article.get('summary', ''),
                                published_at=datetime.fromisoformat(article['time_published']),
                                source=article.get('source', ''),
                                url=article.get('url', ''),
                                sentiment_score=float(article.get('overall_sentiment_score', 0))
                            )
                            news_items.append(news_item)
                        
                        return news_items
                    else:
                        logger.warning(f"Alpha Vantage request failed with status {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return []
    
    async def _fetch_polygon_news(self, symbol: str, days_back: int) -> List[NewsItem]:
        """Fetch news from Polygon."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.polygon.io/v2/reference/news"
                params = {
                    'ticker': symbol,
                    'published_utc.gte': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                    'order': 'desc',
                    'limit': 100,
                    'apiKey': self.api_keys['polygon']
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        news_items = []
                        
                        for article in data.get('results', []):
                            news_item = NewsItem(
                                title=article.get('title', ''),
                                description=article.get('description', ''),
                                content=article.get('description', ''),
                                published_at=datetime.fromisoformat(article['published_utc']),
                                source=article.get('publisher', {}).get('name', ''),
                                url=article.get('article_url', '')
                            )
                            news_items.append(news_item)
                        
                        return news_items
                    else:
                        logger.warning(f"Polygon request failed with status {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching Polygon news: {e}")
            return []
    
    def _deduplicate_news(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate news items based on title similarity."""
        if not news_items:
            return []
        
        # Sort by published date (newest first)
        news_items.sort(key=lambda x: x.published_at, reverse=True)
        
        # Simple deduplication based on title similarity
        unique_news = []
        seen_titles = set()
        
        for item in news_items:
            # Create a simplified title for comparison
            simple_title = re.sub(r'[^\w\s]', '', item.title.lower())
            words = simple_title.split()
            if len(words) > 3:
                title_key = ' '.join(words[:3])  # Use first 3 words as key
            else:
                title_key = simple_title
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(item)
        
        return unique_news

class NLPSignalGenerator:
    """Generates trading signals based on NLP analysis."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.nlp_analyzer = FinancialNLPAnalyzer(config)
        self.news_fetcher = NewsDataFetcher(config)
        
    async def generate_sentiment_signals(self, symbol: str, days_back: int = 7) -> Dict:
        """Generate sentiment-based trading signals."""
        try:
            # Fetch news
            news_items = await self.news_fetcher.fetch_news_async(symbol, days_back)
            
            if not news_items:
                return {
                    'symbol': symbol,
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'signal': 'neutral',
                    'news_count': 0,
                    'recent_news': []
                }
            
            # Analyze sentiment for each news item
            sentiment_scores = []
            confidence_scores = []
            relevant_news = []
            
            for news_item in news_items:
                # Calculate relevance
                relevance = self.nlp_analyzer.calculate_relevance_score(
                    f"{news_item.title} {news_item.description}", symbol
                )
                
                if relevance > 0.3:  # Only consider relevant news
                    # Analyze sentiment
                    sentiment_result = self.nlp_analyzer.analyze_text_sentiment(
                        f"{news_item.title} {news_item.description}"
                    )
                    
                    # Weight by relevance
                    weighted_score = sentiment_result['score'] * relevance
                    weighted_confidence = sentiment_result['confidence'] * relevance
                    
                    sentiment_scores.append(weighted_score)
                    confidence_scores.append(weighted_confidence)
                    
                    # Store relevant news
                    news_item.sentiment_score = sentiment_result['score']
                    news_item.relevance_score = relevance
                    relevant_news.append(news_item)
            
            if not sentiment_scores:
                return {
                    'symbol': symbol,
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'signal': 'neutral',
                    'news_count': len(news_items),
                    'recent_news': []
                }
            
            # Calculate aggregate sentiment
            avg_sentiment = np.mean(sentiment_scores)
            avg_confidence = np.mean(confidence_scores)
            
            # Generate signal
            if avg_sentiment > 0.2 and avg_confidence > 0.5:
                signal = 'buy'
            elif avg_sentiment < -0.2 and avg_confidence > 0.5:
                signal = 'sell'
            else:
                signal = 'hold'
            
            return {
                'symbol': symbol,
                'sentiment_score': avg_sentiment,
                'confidence': avg_confidence,
                'signal': signal,
                'news_count': len(news_items),
                'relevant_news_count': len(relevant_news),
                'recent_news': relevant_news[:5],  # Top 5 most recent
                'sentiment_distribution': {
                    'positive': len([s for s in sentiment_scores if s > 0.1]),
                    'negative': len([s for s in sentiment_scores if s < -0.1]),
                    'neutral': len([s for s in sentiment_scores if -0.1 <= s <= 0.1])
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating sentiment signals for {symbol}: {e}")
            return {
                'symbol': symbol,
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'signal': 'neutral',
                'news_count': 0,
                'recent_news': [],
                'error': str(e)
            }
    
    def analyze_earnings_call_transcript(self, transcript_text: str) -> Dict:
        """Analyze earnings call transcript for sentiment and key insights."""
        if not transcript_text:
            return {}
        
        # Split transcript into sections
        sections = self._split_transcript_sections(transcript_text)
        
        # Analyze each section
        section_analysis = {}
        overall_sentiment = 0.0
        key_insights = []
        
        for section_name, section_text in sections.items():
            if len(section_text) > 50:  # Only analyze substantial sections
                sentiment_result = self.nlp_analyzer.analyze_text_sentiment(section_text)
                key_phrases = self.nlp_analyzer.extract_key_phrases(section_text)
                
                section_analysis[section_name] = {
                    'sentiment': sentiment_result['score'],
                    'confidence': sentiment_result['confidence'],
                    'key_phrases': key_phrases[:5],
                    'text_length': len(section_text)
                }
                
                overall_sentiment += sentiment_result['score']
                key_insights.extend(key_phrases[:3])
        
        # Calculate overall metrics
        num_sections = len([s for s in section_analysis.values() if s['text_length'] > 50])
        if num_sections > 0:
            overall_sentiment /= num_sections
        
        # Remove duplicate insights
        key_insights = list(set(key_insights))[:10]
        
        return {
            'overall_sentiment': overall_sentiment,
            'section_analysis': section_analysis,
            'key_insights': key_insights,
            'total_sections': len(sections),
            'analyzed_sections': num_sections
        }
    
    def _split_transcript_sections(self, transcript: str) -> Dict[str, str]:
        """Split earnings call transcript into sections."""
        sections = {
            'opening_remarks': '',
            'financial_results': '',
            'business_overview': '',
            'guidance': '',
            'qa_session': ''
        }
        
        # Simple keyword-based section detection
        text_lower = transcript.lower()
        
        # Find opening remarks (usually first part)
        if 'good morning' in text_lower or 'good afternoon' in text_lower:
            start_idx = text_lower.find('good morning') if 'good morning' in text_lower else text_lower.find('good afternoon')
            end_idx = text_lower.find('financial results') if 'financial results' in text_lower else len(transcript) // 4
            sections['opening_remarks'] = transcript[start_idx:end_idx]
        
        # Find financial results section
        if 'financial results' in text_lower or 'revenue' in text_lower:
            start_idx = text_lower.find('financial results') if 'financial results' in text_lower else text_lower.find('revenue')
            end_idx = text_lower.find('business overview') if 'business overview' in text_lower else len(transcript) // 2
            sections['financial_results'] = transcript[start_idx:end_idx]
        
        # Find guidance section
        if 'guidance' in text_lower or 'outlook' in text_lower:
            start_idx = text_lower.find('guidance') if 'guidance' in text_lower else text_lower.find('outlook')
            end_idx = text_lower.find('questions') if 'questions' in text_lower else len(transcript)
            sections['guidance'] = transcript[start_idx:end_idx]
        
        # Find Q&A section
        if 'questions' in text_lower or 'q&a' in text_lower:
            start_idx = text_lower.find('questions') if 'questions' in text_lower else text_lower.find('q&a')
            sections['qa_session'] = transcript[start_idx:]
        
        return sections
    
    def generate_sector_sentiment(self, sector_symbols: List[str]) -> Dict:
        """Generate sector-wide sentiment analysis."""
        sector_sentiments = {}
        
        for symbol in sector_symbols:
            try:
                # Get company info for context
                ticker = yf.Ticker(symbol)
                info = ticker.info
                company_name = info.get('longName', symbol)
                
                # Generate sentiment signal
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                sentiment_result = loop.run_until_complete(
                    self.generate_sentiment_signals(symbol, days_back=3)
                )
                loop.close()
                
                sector_sentiments[symbol] = {
                    'company_name': company_name,
                    'sentiment_score': sentiment_result['sentiment_score'],
                    'confidence': sentiment_result['confidence'],
                    'signal': sentiment_result['signal'],
                    'news_count': sentiment_result['news_count']
                }
                
            except Exception as e:
                logger.error(f"Error analyzing sentiment for {symbol}: {e}")
                sector_sentiments[symbol] = {
                    'company_name': symbol,
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'signal': 'neutral',
                    'news_count': 0
                }
        
        # Calculate sector aggregate
        valid_sentiments = [s['sentiment_score'] for s in sector_sentiments.values() 
                           if s['confidence'] > 0.3]
        
        if valid_sentiments:
            sector_avg_sentiment = np.mean(valid_sentiments)
            sector_confidence = np.mean([s['confidence'] for s in sector_sentiments.values()])
        else:
            sector_avg_sentiment = 0.0
            sector_confidence = 0.0
        
        return {
            'sector_sentiments': sector_sentiments,
            'sector_avg_sentiment': sector_avg_sentiment,
            'sector_confidence': sector_confidence,
            'sector_signal': 'buy' if sector_avg_sentiment > 0.2 else 'sell' if sector_avg_sentiment < -0.2 else 'hold'
        } 