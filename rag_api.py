from fastapi import FastAPI
from pydantic import BaseModel
from agenticRetriever import RAGAgent
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware
import argparse
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent_cache = {
        "agent": RAGAgent(verbose=True, numOfContext=3),
        "numOfContext": 3,
    }
    app.agent_cache = agent_cache

    yield

    print("Closing the model")
    del agent_cache


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Updated model to accept two inputs
class Query(BaseModel):
    question: str
    numOfContext: int



def get_agent(numOfContext: int) -> RAGAgent:
    # Reuse the agent if the context number hasn't changed
    if app.agent_cache["numOfContext"] != numOfContext:
        print(f"Changing Number of Context to ={numOfContext}")
        app.agent_cache["agent"].numOfContext = numOfContext
        app.agent_cache["agent"]._setup_retriever()
        app.agent_cache["numOfContext"] = numOfContext

    return app.agent_cache["agent"]


@app.post("/ask")
def ask_rag(query: Query):
    rag_agent = get_agent(query.numOfContext)
    try:
        response, context = rag_agent(query.question)
        return {"response": response, "context": context}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def helloworld():
    return {
        "message" : "working"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reload", default=False, nargs="?")

    args = parser.parse_args()

    if args.reload:
        uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
    else:
        uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, workers=2)

