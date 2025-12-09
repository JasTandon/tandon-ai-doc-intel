import time
import textstat
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from textblob import TextBlob
import numpy as np
from .models import DocumentResult

class AdvancedAnalytics:
    """
    Performs advanced text metrics and Machine Learning analysis 
    to support research-grade reporting.
    """

    def analyze(self, result: DocumentResult) -> DocumentResult:
        """
        Enriches the result with readability scores, ML-based keywords, topic modeling,
        sentiment analysis, and lexical metrics.
        """
        if not result.text:
            return result
            
        blob = TextBlob(result.text)

        # 1. Readability Metrics (Flesch Reading Ease)
        # 90-100 : Very Easy, 0-30 : Very Confusing
        try:
            result.readability_score = textstat.flesch_reading_ease(result.text)
        except:
            result.readability_score = 0.0
            
        # 2. Semantic Analysis (Sentiment & Subjectivity)
        try:
            result.sentiment_polarity = blob.sentiment.polarity
            result.sentiment_subjectivity = blob.sentiment.subjectivity
        except:
            pass
            
        # 3. Lexical Diversity (Type-Token Ratio)
        try:
            words = blob.words
            if len(words) > 0:
                result.lexical_diversity = len(set(words.lower())) / len(words)
        except:
            pass

        # 4. Traditional ML: Keyphrase Extraction (TF-IDF)
        # Contrast this with LLM entities in your paper
        try:
            # Treat sentences as documents for TF-IDF context
            # or just use the whole text. 
            # Better: Use the chunks we already generated.
            corpus = result.chunks if result.chunks else [result.text]
            
            if len(corpus) > 0:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
                tfidf_matrix = vectorizer.fit_transform(corpus)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top features by summing tfidf scores across all chunks
                dense = tfidf_matrix.todense()
                episode = dense.sum(axis=0).tolist()[0]
                phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
                sorted_phrases = sorted(phrase_scores, key=lambda t: t[1] * -1)
                
                top_keywords = []
                for phrase, score in sorted_phrases[:10]:
                    top_keywords.append(feature_names[phrase])
                
                result.ml_keywords = top_keywords
        except Exception as e:
            print(f"TF-IDF failed: {e}")

        # 3. Unsupervised ML: Topic Modeling (NMF)
        # Good for finding "themes" in the chunks
        try:
            if result.chunks and len(result.chunks) > 2:
                # Use CountVectorizer for NMF
                tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
                tfidf = tfidf_vectorizer.fit_transform(result.chunks)
                
                # Fit NMF model (extract 3 topics for this demo)
                n_topics = min(3, len(result.chunks))
                nmf = NMF(n_components=n_topics, random_state=1, l1_ratio=.5, init='nndsvd').fit(tfidf)
                
                feature_names = tfidf_vectorizer.get_feature_names_out()
                found_topics = []
                
                for topic_idx, topic in enumerate(nmf.components_):
                    top_features_ind = topic.argsort()[:-6:-1]
                    topic_words = [feature_names[i] for i in top_features_ind]
                    found_topics.append(f"Topic {topic_idx+1}: {', '.join(topic_words)}")
                
                result.topics = found_topics
        except Exception as e:
            # Often fails if text is too short or vocabulary too small
            # print(f"Topic modeling failed: {e}") 
            pass

        return result

