from openai import OpenAI
from api.core.config import settings
import requests
import json


from openai import OpenAI
from api.core.config import settings
import requests
import json
import time

class HuggingFaceService:
    def __init__(self):
        self.api_key = settings.HUGGINGFACE_API_KEY
        self.model_name = settings.HUGGINGFACE_MODEL
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY is required for Hugging Face provider")

    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return "I don't have enough information to answer this question based on the provided documents."
        
        context_text = "\n\n".join([
            f"[Context {i+1}]: {chunk}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""<s>[INST] <<SYS>>
You are a helpful AI assistant. Use the following context to answer the user's question.
Answer based ONLY on the provided context. If the answer cannot be found in the context, 
say "I don't have enough information to answer that based on the provided documents."
Be concise and factual. Do not make up information.
<</SYS>>

Context Information:
{context_text}

Question: {query}

Answer: [/INST]"""
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.1,
                    "do_sample": False,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').strip()
                return str(result)
            elif response.status_code == 503:
                # Model is loading, wait and retry
                time.sleep(10)
                return self.generate_answer(query, context_chunks)
            else:
                return f"Hugging Face API error (HTTP {response.status_code}): {response.text}"
                
        except Exception as e:
            return f"Error calling Hugging Face API: {str(e)}"

            
class OpenAIService:
    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_name = settings.OPENAI_MODEL

    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        context_text = "\n\n".join([
            f"[Context {i+1}]: {chunk}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question.

Context Information:
{context_text}

Question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. If the answer cannot be found in the context, say "I don't have enough information to answer that based on the provided documents."
3. Be concise and factual
4. Do not make up information

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that answers questions based on provided context."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"

class LocalLLMService:
    def __init__(self):
        self.model_name = settings.LOCAL_LLM_MODEL
        self.api_url = settings.LOCAL_LLM_URL

    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return "I don't have enough information to answer this question based on the provided documents."
        
        context_text = "\n\n".join([
            f"[Context {i+1}]: {chunk}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""Use the following context to answer the question. If you don't know the answer, say so.

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            # For local LLMs using OpenAI-compatible API
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                return f"Local LLM error: {response.text}"
                
        except Exception as e:
            return f"Error calling local LLM: {str(e)}"

class MockLLMService:
    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        """Mock service for testing without any LLM"""
        if not context_chunks:
            return "I don't have enough information to answer this question."
        
        return f"Based on the provided documents, this is a mock response to: {query}. The system found {len(context_chunks)} relevant context chunks."