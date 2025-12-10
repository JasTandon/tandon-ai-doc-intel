import time
import textstat
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from textblob import TextBlob
import numpy as np
import logging
from .models import DocumentResult

logger = logging.getLogger(__name__)

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
            
        logger.info("    [Analytics] Starting analysis...")
        
        # Limit text for expensive operations
        # TextBlob and textstat can be slow on massive texts
        # 50k chars is plenty for these metrics (approx 10-15 pages of text)
        analysis_text = result.text[:50000]
        
        try:
            # Explicitly force download of missing corpora if needed
            try:
                blob = TextBlob(analysis_text)
                # Access a property to trigger lazy loading and potential error
                _ = blob.words 
            except LookupError:
                logger.warning("    [Analytics] Missing NLTK data. Downloading...")
                import nltk
                nltk.download('punkt')
                nltk.download('brown')
                nltk.download('wordnet')
                nltk.download('averaged_perceptron_tagger')
                blob = TextBlob(analysis_text)
        except Exception as e:
            logger.error(f"    [Analytics] TextBlob init failed: {e}")
            blob = None

        # 1. Readability Metrics
        logger.info("    [Analytics] Calculating readability...")
        try:
            result.readability_score = textstat.flesch_reading_ease(analysis_text)
            result.gunning_fog = textstat.gunning_fog(analysis_text)
            result.automated_readability_index = textstat.automated_readability_index(analysis_text)
        except Exception as e:
            logger.warning(f"    [Analytics] Readability metrics failed: {e}")
            result.readability_score = 0.0
            result.gunning_fog = 0.0
            result.automated_readability_index = 0.0
            
        # 2. Semantic Analysis (Sentiment & Subjectivity)
        if blob:
            logger.info("    [Analytics] Calculating sentiment...")
            try:
                result.sentiment_polarity = blob.sentiment.polarity
                result.sentiment_subjectivity = blob.sentiment.subjectivity
            except Exception as e:
                logger.warning(f"    [Analytics] Sentiment analysis failed: {e}")
            
            # 3. Lexical Diversity (Type-Token Ratio)
            logger.info("    [Analytics] Calculating lexical diversity...")
            try:
                words = blob.words
                if len(words) > 0:
                    result.lexical_diversity = len(set(words.lower())) / len(words)
            except Exception as e:
                logger.warning(f"    [Analytics] Lexical diversity failed: {e}")

        # 4. Traditional ML: Keyphrase Extraction (TF-IDF)
        logger.info("    [Analytics] Extracting keyphrases (TF-IDF)...")
        try:
            corpus = result.chunks if result.chunks else [result.text[:50000]]
            
            if len(corpus) > 0:
                # Limit features to speed up
                vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
                tfidf_matrix = vectorizer.fit_transform(corpus)
                feature_names = vectorizer.get_feature_names_out()
                
                dense = tfidf_matrix.todense()
                episode = dense.sum(axis=0).tolist()[0]
                phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
                sorted_phrases = sorted(phrase_scores, key=lambda t: t[1] * -1)
                
                top_keywords = []
                for phrase, score in sorted_phrases[:10]:
                    top_keywords.append(feature_names[phrase])
                
                result.ml_keywords = top_keywords
        except Exception as e:
            logger.warning(f"    [Analytics] TF-IDF failed: {e}")

        # 5. Unsupervised ML: Topic Modeling (NMF)
        logger.info("    [Analytics] Modeling topics (NMF)...")
        try:
            # Check if we have enough chunks
            if result.chunks and len(result.chunks) > 2:
                # Use TfidfVectorizer for NMF input
                tfidf_vectorizer = TfidfVectorizer(
                    max_df=0.95, 
                    min_df=2, 
                    stop_words='english', 
                    max_features=500 # Drastically reduce features to avoid hangs
                )
                tfidf = tfidf_vectorizer.fit_transform(result.chunks)
                
                n_topics = min(3, len(result.chunks))
                
                if tfidf.shape[0] >= n_topics and tfidf.shape[1] >= n_topics:
                    # Use 'random' init for maximum stability if nndsvd is hanging
                    # Also use 'cd' solver which can be more robust than 'mu'
                    nmf = NMF(
                        n_components=n_topics, 
                        random_state=42, 
                        init='random',
                        solver='cd',
                        max_iter=200 
                    )
                    # Fit in a way that respects timeouts?
                    # Sklearn doesn't support timeout natively.
                    # We can rely on the drastic reduction in max_features (1000 -> 500)
                    nmf.fit(tfidf)
                    
                    feature_names = tfidf_vectorizer.get_feature_names_out()
                    found_topics = []
                    
                    for topic_idx, topic in enumerate(nmf.components_):
                        top_features_ind = topic.argsort()[:-6:-1]
                        topic_words = [feature_names[i] for i in top_features_ind]
                        found_topics.append(f"Topic {topic_idx+1}: {', '.join(topic_words)}")
                    
                    result.topics = found_topics
                else:
                    logger.info("    [Analytics] Not enough data/features for NMF.")
            else:
                logger.info("    [Analytics] Not enough chunks for NMF.")
                
        except Exception as e:
            logger.warning(f"    [Analytics] Topic modeling failed: {e}")

        # 6. Technical Metrics (Information Density, Sentence Complexity)
        if blob:
            try:
                # Information Density: Ratio of non-stop words to total words
                # Approximation using simple length check if stopwords not loaded
                # Let's use word length > 3 as simple heuristic for "content" words
                total_words = len(blob.words)
                if total_words > 0:
                    long_words = [w for w in blob.words if len(w) > 3]
                    result.info_density = len(long_words) / total_words
                
                # Entity Density: We need entities from result.entities
                if result.entities and total_words > 0:
                    result.entity_density = len(result.entities) / total_words
                
                # Sentence Complexity: Std Dev of sentence length
                sentences = blob.sentences
                if len(sentences) > 1:
                    lengths = [len(s.words) for s in sentences]
                    result.sentence_complexity = float(np.std(lengths))
                elif len(sentences) == 1:
                    result.sentence_complexity = 0.0
                    
            except Exception as e:
                logger.warning(f"    [Analytics] Technical metrics failed: {e}")

        logger.info("    [Analytics] Analysis complete.")
        return result

