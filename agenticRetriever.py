"""
RAG Retrieval Agent Application

This module implements a Retrieval-Augmented Generation (RAG) system using LangChain and LangGraph.
The system can:
1. Load and access a locally stored vector database
2. Process user queries through an agent that decides whether to retrieve documents
3. Grade document relevance and optionally rewrite the query
4. Generate final responses based on retrieved documents

Usage:
    from rag_agent import RAGAgent

    # Initialize the agent
    rag_agent = RAGAgent()

    # Process a query
    response = rag_agent.process_query("What is strong AI?")
"""

import os
import warnings
from typing import Annotated, Dict, List, Literal, Sequence, Any, Optional

# Suppress warning messages
warnings.filterwarnings("ignore")

# Load environment variables
from dotenv import load_dotenv

# OpenAI client and Pydantic base classes
from openai import OpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# LangChain components
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolCall
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool

# LangGraph components
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
import re
from keywords import KeywordExtractor
from creatingVectorDB import VectorDatabaseManager
from creatingVectorDB import VectorDatabaseManager

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Directory for persisting the vector database
PERSIST_DIRECTORY = "C:\\Users\\hasee\\Desktop\\NCAI\\Codes\\DomainSpecificChatbotWebAppBackend\\DomainSpecificChatbotWebAppBackend\\VectorDBs"
finalContext = []

MODEL_NAMES = {
    "grade_documents" : "qwen2.5:3b",
    "rewrite" : "qwen2.5:3b",
    "generate" : "qwen2.5:3b",

}

EMBEDDING_MODEL = "BAAI/bge-small-en"


class AgentState(TypedDict):
    """
    Type definition for the agent's state.
    
    Attributes:
        messages: Sequence of conversation messages
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


class LoggingRetriever(BaseRetriever):
    """
    Custom retriever that logs retrieved documents and filters duplicates.

    Attributes:
        base_retriever: The base retriever to delegate retrieval to
        seen_hashes: A set to track already seen document hashes
    """
    base_retriever: BaseRetriever = Field(...)
    seen_hashes: set = Field(default_factory=set)

    def _get_relevant_documents(self, query, *, run_manager=None):
        """
        Retrieve and log relevant documents for a given query.

        Args:
            query: User input query
            run_manager: Optional run manager for logging

        Returns:
            List of unique and relevant documents
        """
        # self.seen_hashes.clear()
        print("line 105")
        docs = self.base_retriever._get_relevant_documents(query, run_manager=run_manager)
        print(f"docs at line 106 {docs=}")
        unique_docs = []
        docs_with_metadata = []
        for doc in docs:
            # Create a unique hash using content and metadata
            doc_hash = hash(f"{doc.page_content}-{doc.metadata}")
            print("Document Hash:", doc_hash)
            if doc_hash not in self.seen_hashes:
                print("Document is unique, adding to the list.")
                self.seen_hashes.add(doc_hash)
                unique_docs.append(doc)

                source = os.path.basename(doc.metadata.get('source', 'unknown'))
                page = doc.metadata.get('page', 'unknown')
                docs_with_metadata.append({"doc": doc, "source": source, "page": page})

                print(f"Retrieved: {source} p.{page} - \n{doc.page_content=}")

        global finalContext 
        finalContext = docs_with_metadata

        return unique_docs


class RAGAgent:
    """
    RAG (Retrieval-Augmented Generation) Agent class that handles query processing.
    """
    
    def __init__(self, verbose: bool = True, numOfContext=3):
        """
        Initialize the agent and set up its components.

        Args:
            verbose: Whether to print debug information
            numOfContext: Number of documents to retrieve for context
        """
        self.verbose = verbose
        # self.client = OpenAI()
        self.numOfContext = numOfContext
        self.context = []
        self.context_failure = 0 
        self.embeddingModel = EMBEDDING_MODEL

        self._load_vector_db()
        self._setup_retriever()
        self._build_workflow()
        self.keywordExtractor = KeywordExtractor()  # Placeholder for keyword extractor if needed

    def _load_vector_db(self):
        """Load vector database from disk."""
        if self.verbose:
            print(f"Loading vector database from {PERSIST_DIRECTORY}")
        
        embedding_function = HuggingFaceEmbeddings(model_name=self.embeddingModel)
        print(f"{embedding_function=}")
        print(f"{PERSIST_DIRECTORY=}")
        self.vectorstore = Chroma(
            collection_name="rag-chroma",
            embedding_function=embedding_function,
            persist_directory=PERSIST_DIRECTORY
        )

        if self.verbose:
            print("Vector database loaded successfully")

    def _setup_retriever(self):
        """Configure retriever with Maximal Marginal Relevance and logging."""
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.numOfContext,
                "fetch_k": 20,
                "lambda_mult": 0.5
            }
        )

        self.logging_retriever = LoggingRetriever(base_retriever=retriever)

        self.vectorDBManager = VectorDatabaseManager(documents_directory="C:\\Users\\hasee\\Desktop\\NCAI\\Codes\\DomainSpecificChatbotWebAppBackend\\3books", model_name=EMBEDDING_MODEL, collection_name="rag-chroma")

    def _build_workflow(self):
        """Build the LangGraph workflow for query handling."""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_node("rewrite", self._rewrite)

        # Define edges
        workflow.add_edge(START, "retrieve")

        workflow.add_conditional_edges(
            "retrieve",
            self._grade_documents,
            {
                "generate": "generate",
                "rewrite": "rewrite",
            }
        )

        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "retrieve")

        self.graph = workflow.compile()

    def _retrieve(self, state):
        query = state["messages"][0].content
        keywords = self.keywordExtractor.extract_keywords(query)

        keyword_extractions = {}
        combined_text = ""
        self.context = []


        keyword_extractions = {}
        combined_text = ""

        for key, value in keywords.items():
            print(f"{key=}")
            # docs = self.logging_retriever._get_relevant_documents(key)
            docs = self.vectorDBManager.search_documents(key, k=self.numOfContext)
            print(f"{docs=}")
            keyword_extractions[key] = docs
            combined_text += "\n\n".join(doc.page_content for doc in docs)
            for doc in docs:
                source = os.path.basename(doc.metadata.get('source', 'unknown'))
                page = doc.metadata.get('page', 'unknown')
                self.context.append({"doc": doc, "source": source, "page": page})

        print("Extracted Keywords:", keywords.keys())
        print(f"{combined_text=}")

            print(f"{key=}")
            # docs = self.logging_retriever._get_relevant_documents(key)
            docs = self.vectorDBManager.search_documents(key, k=self.numOfContext)
            print(f"{docs=}")
            keyword_extractions[key] = docs
            combined_text += "\n\n".join(doc.page_content for doc in docs)

        print("Extracted Keywords:", keywords.keys())
        print(f"{combined_text=}")

        return {"messages": [AIMessage(content=combined_text)]}


    def _grade_documents(self, state) -> Literal["generate", "rewrite"]:
        """
        Decide if retrieved documents are relevant enough.

        Args:
            state: Current state including query and context

        Returns:
            A string to determine the next step in the workflow
        """
        if self.verbose:
            print("---CHECK RELEVANCE---")

        class Grade(BaseModel):
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
        model = ChatOllama(
            model=MODEL_NAMES["grade_documents"],
            temperature=1
        )
        llm_with_tool = model.with_structured_output(Grade)

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the question can be answered using the document.""",
            input_variables=["context", "question"],
        )

        chain = prompt | llm_with_tool

        messages = state["messages"]
        last_message = messages[-1]
        question = messages[0].content
        docs = last_message.content

        scored_result = chain.invoke({"question": question, "context": docs})
        print(f"Scored Result: {scored_result}")
        score = scored_result.binary_score

        if score == "yes":
            global finalContext
            # self.context = finalContext
            if self.verbose:
                print("---DECISION: DOCS RELEVANT---")
            return "generate"
        else:
            if self.verbose:
                print("---DECISION: DOCS NOT RELEVANT---")
            self.context_failure +=1
            if self.context_failure >=2:
                return "generate"
            return "rewrite"

    def _rewrite(self, state):
        """
        Rewrite the user query to improve retrieval results.

        Args:
            state: Current state containing the original query

        Returns:
            Updated message list with rewritten query
        """
        if self.verbose:
            print("---TRANSFORM QUERY---")

        question = state["messages"][0].content
        #Look at the input and try to reason about the underlying semantic intent / meaning. \n
        msg = [
            HumanMessage(
                content=f""" \n  
        #Look at the input and try to reason about the underlying semantic intent / meaning. \n
        Do not answer the question. Only output an imporved question.\n
        Here is the initial question. :
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """
            )
        ]

        # model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
        model = ChatOllama(
            model=MODEL_NAMES["rewrite"],
            temperature=0
        )
        response = model.invoke(msg)
        response.content = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()


        return {"messages": [response]}

    def _generate(self, state):
        """
        Generate the final answer based on retrieved documents.

        Args:
            state: Current state containing user query and context

        Returns:
            Message with the final generated response
        """
        if self.verbose:
            print("---GENERATE---")
        
        if self.context_failure >=2:
            # return {"messages": [AIMessage(content="This question is out of my scope.")]}
            self.context = []
            return {"messages": ["This question is out of my scope."]}

        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = hub.pull("rlm/rag-prompt")
        # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
        # llm = OllamaChat()
        llm = ChatOllama(
            model=MODEL_NAMES["generate"],
            temperature=0)
        # llm.streaming = True

        formatted_prompt = prompt.format(context=docs, question=question)

        if self.verbose:
            print("Final Prompt:\n", formatted_prompt)

        rag_chain = prompt | llm | StrOutputParser()
        response = rag_chain.invoke({"context": docs, "question": question})

        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        #Empty seen hashes
        self.logging_retriever.seen_hashes.clear()
        
        return {"messages": [response]}

    def __call__(self, query: str) -> str:
        """
        Run the RAG agent end-to-end with the provided query.

        Args:
            query: Input question from the user

        Returns:
            Response string and list of context metadata
        """
        inputs = {
            "messages": [
                HumanMessage(content=query)
            ]
        }

        result = None
        self.context_failure = 0
        self.logging_retriever.seen_hashes.clear()

        for output in self.graph.stream(inputs):
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print(value)
                print("---\n")
                result = value

        extracted_data = []
        print(f"{extracted_data=}")
        for entry in self.context:
            doc = entry.get("doc", {})
            extracted_data.append({
                "book_title": os.path.basename(doc.metadata["source"]).split(".")[0],
                "page_number": doc.metadata["page"],
                "page_content": doc.page_content
            })

        return result["messages"][0], extracted_data


# Example usage
if __name__ == "__main__":
    rag_agent = RAGAgent(verbose=True)
    response, context = rag_agent("What is an RNN?")
    print("\nFinal Response:\n", response)
    print("Final Context:", context)
