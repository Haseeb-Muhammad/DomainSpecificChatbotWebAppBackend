from fastapi import FastAPI
from pydantic import BaseModel
from agenticRetrieverv4 import RAGAgent
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import argparse
from contextlib import asynccontextmanager
from creatingVectorDB import VectorDatabaseManager 
from fastapi import UploadFile, File
import shutil
import os
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from threading import Lock
import time


job_status = {
    "state": "idle",  # or: processing, success, failed
    "message": "No processing job running.",
}
job_lock = Lock()

documents_dir = "/home/haseebmuhammad/Desktop/AITeacherChatbot/CQADatasetFromBooks/AI-books"
db_manager = VectorDatabaseManager(documents_directory=documents_dir)

@asynccontextmanager
async def lifespan(app: FastAPI):
    agent_cache = {
        "agent": RAGAgent(db_menager=db_manager,verbose=True, numOfContext=3),
        "numOfContext": 3,
    }
    app.agent_cache = agent_cache

    yield

    print("Closing the model")
    del agent_cache


app = FastAPI(lifespan=lifespan)
# app = FastAPI()

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
        # app.agent_cache["agent"]._setup_retriever()
        app.agent_cache["numOfContext"] = numOfContext

    return app.agent_cache["agent"]


@app.post("/ask")
def ask_rag(query: Query):
    start = time.time()
    rag_agent = get_agent(query.numOfContext)
    try:
        response, context = rag_agent(query.question)
        end = time.time()
        print(f"That query took {end-start} seconds")
        return {"response": str(response), "context": context}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def helloworld():
    return {
        "message" : "working"
    }

@app.get("/pdfs")
def list_pdfs():
    pdfs = []
    for filename in os.listdir(db_manager.documents_directory):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(db_manager.documents_directory, filename)
            size_bytes = round(os.path.getsize(path), 2)
            upload_time = datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
            pdfs.append({
                "filename": filename,
                "size_bytes": size_bytes,
                "upload_time": upload_time
            })
    return {"pdfs": pdfs}

@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    # Only allow PDF uploads
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are allowed."}
    
    save_path = os.path.join(db_manager.documents_directory, file.filename)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # db_manager.create_or_update_database()
    
    return {"message": f"Uploaded {file.filename} successfully."}



@app.delete("/delete-pdf/{filename}")
def delete_pdf(filename: str):
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    
    file_path = os.path.join(db_manager.documents_directory, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF file not found in directory.")

    # Delete the PDF file
    os.remove(file_path)

    # Delete vectors from database
    try:
        deleted_count = db_manager.delete_documents_by_pdf_name(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting vectors: {str(e)}")

    return {
        "message": f"Deleted {filename} from directory.",
        "vectors_deleted": deleted_count
    }


@app.get("/database/pdfs")
def list_pdfs():
    return list(db_manager.list_pdf_names_in_database())

def run_job(new_only: bool, limit: int):
    try:
        db_manager.create_or_update_database(new_pdfs_only=new_only, pdf_limit=limit)
        # time.sleep(10)
        job_status["state"] = "success"
        job_status["message"] = "Processing has been completed successfully."
    except Exception as e:
        job_status["state"] = "failed"
        job_status["message"] = str(e)

@app.post("/database/update")
def create_or_update(new_only: bool = True, limit: int = None, background_tasks: BackgroundTasks = None):
    if job_status["state"] == "processing":
        return {"state": "processing", "message": "Previous job still running."}

    job_status["state"] = "processing"
    job_status["message"] = "Processing job running."
    background_tasks.add_task(run_job, new_only, limit)
    return {"state": "queued", "message": "Processing started in background."}

@app.get("/database/job-status")
def get_status():
    return job_status

# KEEP IT COMMENTED OUT 
# @app.post("/database/update")
# def create_or_update(new_only: bool = True, limit: int = None):
#     print(f"create_or_update called with new_only={new_only}, limit={limit}")
#     db = db_manager.create_or_update_database(new_pdfs_only=new_only, pdf_limit=limit)
#     return db

class SearchRequest(BaseModel):
    query: str
    k: int = 5

@app.post("/search")
def search_docs(body: SearchRequest):
    results = db_manager.search_documents(body.query, k=body.k)
    return {"results": [r.page_content for r in results]}


@app.delete("/database/pdf/{pdf_name}")
def delete_pdf(pdf_name: str):
    count = db_manager.delete_documents_by_pdf_name(pdf_name)
    return {"deleted": count}


@app.get("/database/stats")
def get_stats():
    return db_manager.get_database_statistics()


@app.delete("/database/reset")
def reset_db():
    success = db_manager.reset_database()
    return {"reset": success}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reload", default=False, nargs="?")

    args = parser.parse_args()

    if args.reload:
        uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
    else:
        uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, workers=2)
