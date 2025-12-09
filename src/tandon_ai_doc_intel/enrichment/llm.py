import os
from typing import List, Dict, Any, Optional
from openai import OpenAI

class LLMEnricher:
    """
    Enriches document text using LLM (e.g., OpenAI).
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # We don't raise here to allow instantiation, but methods will fail or return empty
            pass
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = model

    def chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """
        Splits text into chunks with overlap.
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
            
        return chunks

    def summarize(self, text: str) -> str:
        """
        Generates a summary of the text.
        """
        if not self.client or not text:
            return "LLM not configured or empty text."
            
        # For long text, we might need to summarize chunks and then map-reduce, 
        # but for this MVP we'll take the first chunk or truncate.
        # A real production system would be more sophisticated.
        truncated_text = text[:4000] # Simple truncation to fit context
        
        prompt = f"Please provide a concise summary of the following text:\n\n{truncated_text}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating summary: {e}"

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts named entities (people, organizations, dates, locations).
        """
        if not self.client or not text:
            return []
            
        truncated_text = text[:4000]
        prompt = (
            "Extract key entities (Person, Organization, Location, Date) from the text below. "
            "Return the result as a JSON list of objects with 'name' and 'type' keys.\n\n"
            f"{truncated_text}"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            import json
            content = response.choices[0].message.content
            # Expecting JSON output like { "entities": [...] } or just [...]
            data = json.loads(content)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "entities" in data:
                return data["entities"]
            return []
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []

    def analyze_risk(self, text: str) -> Dict[str, Any]:
        """
        Analyzes potential risks in the document.
        """
        if not self.client or not text:
            return {}
            
        truncated_text = text[:4000]
        prompt = (
            "Analyze the following text for potential legal, financial, or compliance risks. "
            "Return a JSON object with 'risk_level' (Low/Medium/High) and 'risk_factors' (list of strings).\n\n"
            f"{truncated_text}"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            import json
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error analyzing risk: {e}")
            return {"error": str(e)}

