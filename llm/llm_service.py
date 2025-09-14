import os
from openai import OpenAI

class LLMService:
    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        """
        Build a prompt with retrieved context and query, then call the LLM.
        """
        context_text = "\n\n".join(context_chunks)

        prompt = f"""You are an AI assistant. Use the following context to answer the question.
If the answer is not found in the context, say "I don't know based on the provided documents."

Context:
{context_text}

Question:
{query}

Answer:"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # lower = more factual
            max_tokens=500
        )

        return response.choices[0].message.content.strip()
