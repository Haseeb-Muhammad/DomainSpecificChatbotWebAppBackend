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

@app.post("/ask")
def ask_rag(query: Query):
    rag_agent = RAGAgent(verbose=True, numOfContext=query.numOfContext)
    try:
        response, context = rag_agent(query.question)
        extracted_data = []
        for entry in context:
            doc = entry.get("doc", {})

            extracted_data.append({
                "book_title": os.path.basename(doc.metadata["source"]).split(".")[0],
                "page_number": doc.metadata["page"],
                "page_content": doc.page_content
            })
        return {"response": response, "context": extracted_data}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
