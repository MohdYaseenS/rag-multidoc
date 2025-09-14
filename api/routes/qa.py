from fastapi import APIRouter
from pydantic import BaseModel
from retrieval.vector_store import VectorStore
from api.core.config import settings
from openai import OpenAI

router = APIRouter()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

vs = VectorStore()

class AskRequest(BaseModel):
    query: str
    top_k: int = 3

class AskResponse(BaseModel):
    answer: str
    context: list

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Step 1: retrieve top chunks
    results = vs.search(req.query, top_k=req.top_k)
    context_texts = [c["text"] for c, _ in results]

    # Step 2: construct prompt
    context_str = "\n\n".join(context_texts)
    prompt = f"""You are a helpful assistant. 
Use the following context to answer the question.

Context:
{context_str}

Question: {req.query}
Answer:"""

    # Step 3: call LLM (GPT-4o-mini here)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI answering questions based on provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    return AskResponse(answer=answer, context=context_texts)