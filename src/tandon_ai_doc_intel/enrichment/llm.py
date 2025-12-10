import diskcache
import hashlib
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class LLMEnricher:
    """
    Enriches text using LLMs (Summarization, Entity Extraction, Risk Analysis).
    """
    
    def __init__(self, api_key: Optional[str] = None, use_caching: bool = True):
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.cache = diskcache.Cache("./llm_cache") if use_caching else None
        
        # Token Tracking
        self.token_usage = {"llm_input": 0, "llm_output": 0}

    def _get_cache_key(self, prefix: str, text: str) -> str:
        """Generates a SHA-256 cache key."""
        hash_digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return f"{prefix}:{hash_digest}"

    def chunk_text(self, text: str, max_tokens: int = 500) -> List[str]:
        """
        Splits text into manageable chunks.
        (Simple character-based splitting for now, could use tiktoken).
        """
        if not text:
            return []
            
        # Approx 4 chars per token
        chunk_size = max_tokens * 4
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        return chunks

    def summarize(self, text: str) -> str:
        """
        Generates a summary of the document.
        """
        if not self.client:
            return "Summary not available (No API Key)"
            
        truncated_text = text[:10000] # Limit context
        
        if self.cache:
            key = self._get_cache_key("summary", truncated_text)
            cached = self.cache.get(key)
            if cached:
                logger.info("Enrichment: Using cached summary.")
                # We don't track tokens for cached hits? Or should we estimate saved tokens?
                # For cost estimation, we shouldn't count them.
                return cached

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Summarize the following document concisely."},
                    {"role": "user", "content": truncated_text}
                ],
                max_tokens=300
            )
            
            # Track Usage
            if response.usage:
                self.token_usage["llm_input"] += response.usage.prompt_tokens
                self.token_usage["llm_output"] += response.usage.completion_tokens
            
            summary = response.choices[0].message.content
            
            if self.cache:
                self.cache.set(key, summary)
                
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "Summary generation failed."

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts key entities (People, Orgs, Dates).
        """
        if not self.client:
            return []

        truncated_text = text[:5000] 
        
        if self.cache:
            key = self._get_cache_key("entities", truncated_text)
            cached = self.cache.get(key)
            if cached:
                return cached

        try:
            prompt = """
            Extract key entities from the text below. 
            Return a Python list of dictionaries like [{"text": "Entity Name", "label": "ORG/PERSON/DATE"}].
            Only return the list, no other text.
            """
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": truncated_text}
                ],
                temperature=0
            )
            
            # Track Usage
            if response.usage:
                self.token_usage["llm_input"] += response.usage.prompt_tokens
                self.token_usage["llm_output"] += response.usage.completion_tokens

            content = response.choices[0].message.content
            
            # Safe eval
            import ast
            try:
                entities = ast.literal_eval(content)
                if not isinstance(entities, list):
                    entities = []
            except:
                entities = [{"text": content, "label": "RAW_OUTPUT"}]
            
            if self.cache:
                self.cache.set(key, entities)
                
            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    def analyze_risk(self, text: str) -> Dict[str, Any]:
        """
        Analyzes legal/financial risks in the document.
        """
        if not self.client:
            return {"risk_level": "Unknown", "risk_factors": []}

        truncated_text = text[:8000]
        
        if self.cache:
            key = self._get_cache_key("risk", truncated_text)
            cached = self.cache.get(key)
            if cached:
                return cached

        try:
            prompt = """
            Analyze the following text for potential legal, financial, or compliance risks.
            Return a valid Python dictionary (and nothing else) with keys:
            - "risk_level": "Low", "Medium", or "High"
            - "risk_factors": List[str] of specific risks identified.
            """
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": truncated_text}
                ],
                temperature=0
            )
            
            # Track Usage
            if response.usage:
                self.token_usage["llm_input"] += response.usage.prompt_tokens
                self.token_usage["llm_output"] += response.usage.completion_tokens

            content = response.choices[0].message.content
            import ast
            try:
                # Find the dict part if wrapped in markdown
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    risk_data = ast.literal_eval(content[start:end])
                else:
                     risk_data = {"risk_level": "Unknown", "risk_factors": ["Could not parse LLM output"]}
            except:
                risk_data = {"risk_level": "Unknown", "risk_factors": ["Parsing error"]}

            if self.cache:
                self.cache.set(key, risk_data)

            return risk_data
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {"risk_level": "Unknown", "risk_factors": []}
