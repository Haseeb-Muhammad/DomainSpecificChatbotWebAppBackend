# Configuration Guide

## Overview
The codebase now uses a centralized configuration file (`config.py`) that contains all hyperparameters, paths, and settings. This allows you to easily modify the behavior of your RAG system without editing multiple files.

## Configuration File Structure

### Paths
- `DOCUMENTS_DIR`: Directory containing your PDF documents
- `VDB_DIR`: Directory where vector databases are stored
- `LOG_FILE`: Path to the log file

### Vector Database Settings
- `COLLECTION_NAME`: Name of the Chroma collection
- `CHUNK_SIZE`: Size of text chunks for document splitting
- `CHUNK_OVERLAP`: Overlap between chunks

### LLM Models
- `MODEL_NAMES`: Dictionary containing model names for different tasks:
  - `domain_check`: Model for checking if query is ML-related
  - `context_selection`: Model for selecting relevant contexts
  - `grade_documents`: Model for grading document relevance
  - `rewrite`: Model for query rewriting
  - `generate`: Model for final answer generation
- `EMBEDDING_MODEL`: Model for creating embeddings

### RAG Agent Parameters
- `NUM_OF_CONTEXT`: Number of contexts to use for generation
- `MAX_REWRITES`: Maximum number of query rewrites
- `VERBOSE`: Enable/disable verbose logging

### API Settings
- `ALLOWED_ORIGINS`: CORS allowed origins
- `API_HOST`: API server host
- `API_PORT`: API server port

### Processing Parameters
- `BATCH_SIZE`: Batch size for document processing
- `TEMPERATURE_*`: Temperature settings for each model
- `KEYWORD_IMPORTANCE_THRESHOLD`: Minimum importance for keyword extraction
- `DEFAULT_RETRIEVAL_K`: Number of documents to retrieve initially

## How to Use

1. **Modify Configuration**: Edit `config.py` to change any settings
2. **Update Paths**: Change `DOCUMENTS_DIR` and `VDB_DIR` to your desired locations
3. **Change Models**: Update `MODEL_NAMES` and `EMBEDDING_MODEL` to use different models
4. **Adjust Parameters**: Modify chunk sizes, context numbers, temperatures, etc.
5. **Run Your System**: The updated configuration will be automatically used

## Example Modifications

### Change Models
```python
MODEL_NAMES = {
    "domain_check": "llama3.1:8b",
    "context_selection": "llama3.1:8b", 
    "grade_documents": "llama3.1:8b",
    "rewrite": "llama3.1:8b",
    "generate": "llama3.1:8b",
}
EMBEDDING_MODEL = "BAAI/bge-large-en"
```

### Adjust Chunk Sizes
```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
```

### Change Context Numbers
```python
NUM_OF_CONTEXT = 5
DEFAULT_RETRIEVAL_K = 20
```

## Benefits

1. **Centralized Management**: All settings in one place
2. **Easy Experimentation**: Quickly test different configurations
3. **Version Control**: Track configuration changes
4. **Environment-Specific**: Different configs for dev/prod
5. **No Code Changes**: Modify behavior without touching core code

## Files Updated

- `agenticRetrieverv4.py`: Uses config for models, temperatures, and parameters
- `creatingVectorDB.py`: Uses config for embedding model and chunk settings
- `rag_api.py`: Uses config for API settings and database configuration
- `config.py`: New centralized configuration file
