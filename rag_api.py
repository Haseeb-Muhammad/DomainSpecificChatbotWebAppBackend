from fastapi import FastAPI
from pydantic import BaseModel
from agenticRetriever import RAGAgent
import uvicorn
import os
app = FastAPI()

# Updated model to accept two inputs
class Query(BaseModel):
    question: str
    numOfContext: int

# Simple agent cache
agent_cache = {
    "agent": None,
    "numOfContext": None,
}

def get_agent(numOfContext: int) -> RAGAgent:
    # Reuse the agent if the context number hasn't changed
    if agent_cache["agent"] is None or agent_cache["numOfContext"] != numOfContext:
        print(f"Creating new agent with numOfContext={numOfContext}")
        agent_cache["agent"] = RAGAgent(verbose=True, numOfContext=numOfContext)
        agent_cache["numOfContext"] = numOfContext
    return agent_cache["agent"]

@app.post("/ask")
def ask_rag(query: Query):
    rag_agent = get_agent(query.numOfContext)
    try:
        response, context = rag_agent(query.question)
        return {"response": response, "context": context}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
