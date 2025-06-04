from typing import List, Dict, Tuple
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from langchain.schema import Document
import numpy as np

class KeywordExtractor:
    def __init__(self, model_name: str = 'all-MiniLM-L12-v2'):
        """
        Initialize the keyword extractor with a KeyBERT model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.kw_model = KeyBERT(model=model_name)
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing and removing stopwords and punctuation.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            List[str]: List of preprocessed tokens
        """
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in string.punctuation]
        tokens = [t for t in tokens if t not in stopwords.words('english')]
        return tokens

    def extract_keywords(self, 
                        text: str, 
                        top_n: int = 5, 
                        mmr_diversity: float = 0.7,
                        min_score: float = 0.6) -> Dict[str, float]:
        """
        Extract keywords from text using KeyBERT.
        
        Args:
            text (str): Input text to extract keywords from
            top_n (int): Number of keywords to extract
            mmr_diversity (float): Diversity parameter for MMR
            min_score (float): Minimum score threshold for keywords
            
        Returns:
            Dict[str, float]: Dictionary of keywords and their scores
        """
        keyphrases = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            use_mmr=True,
            diversity=mmr_diversity,
            top_n=top_n
        )
        return {phrase: score for phrase, score in keyphrases if score >= min_score}

class KeywordDatabase:
    def __init__(self, db_path: str):
        """
        Initialize the keyword database.
        
        Args:
            db_path (str): Path to the JSON file containing the keyword database
        """
        self.db_path = db_path
        self.keyword_db = self.load_database()

    def load_database(self) -> Dict:
        """
        Load the keyword database from JSON file.
        
        Returns:
            Dict: Loaded keyword database
        """
        with open(self.db_path, "r") as f:
            return json.load(f)

    def get_document_keywords(self, doc_id: str) -> List[Tuple[str, float]]:
        """
        Get keywords for a specific document.
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            List[Tuple[str, float]]: List of (keyword, score) tuples
        """
        return self.keyword_db.get(doc_id, [])

class DocumentReranker:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the document reranker with a sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.embedder = SentenceTransformer(model_name)

    def compute_overlap_score(self, 
                            query_keywords: List[str], 
                            doc_keywords: List[Tuple[str, float]]) -> float:
        """
        Compute overlap score between query keywords and document keywords.
        
        Args:
            query_keywords (List[str]): List of query keywords
            doc_keywords (List[Tuple[str, float]]): List of (keyword, score) tuples from document
            
        Returns:
            float: Overlap score
        """
        doc_kw_dict = {kw: score for kw, score in doc_keywords}
        return sum(doc_kw_dict[kw] for kw in query_keywords if kw in doc_kw_dict)

    def rerank_documents(self,
                        retrieved_docs: List[Document],
                        query_keywords: List[str],
                        keyword_db: KeywordDatabase,
                        top_k_similarities: int = 3) -> List[Document]:
        """
        Rerank documents based on semantic similarity of keywords.
        
        Args:
            retrieved_docs (List[Document]): List of retrieved documents
            query_keywords (List[str]): List of query keywords
            keyword_db (KeywordDatabase): Keyword database instance
            top_k_similarities (int): Number of top similarities to consider
            
        Returns:
            List[Document]: Reranked documents
        """
        # Embed query keywords
        query_embeddings = self.embedder.encode(query_keywords, convert_to_tensor=True)
        doc_scores = []
        
        for doc in retrieved_docs:
            doc_id = doc.metadata.get("doc_id")
            if not doc_id or doc_id not in keyword_db.keyword_db:
                doc_scores.append((doc, 0.0))
                continue

            # Get document keywords
            doc_keywords = [kw for kw, score in keyword_db.get_document_keywords(doc_id)]
            if not doc_keywords:
                doc_scores.append((doc, 0.0))
                continue

            # Compute similarities
            doc_embeddings = self.embedder.encode(doc_keywords, convert_to_tensor=True)
            cosine_sim_matrix = util.cos_sim(query_embeddings, doc_embeddings)
            
            # Aggregate scores
            max_similarities = cosine_sim_matrix.max(dim=1).values
            score = max_similarities.topk(min(top_k_similarities, len(max_similarities))).values.mean().item()
            
            doc_scores.append((doc, score))

        # Sort by descending score
        ranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs] 