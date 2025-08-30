import os
import warnings
from typing import Annotated, Dict, List, Literal, Sequence, Any, Optional

# Suppress warning messages
warnings.filterwarnings("ignore")

# Load environment variables
from dotenv import load_dotenv

# OpenAI client and Pydantic base classes
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# LangChain components
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolCall
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from langchain.tools.retriever import create_retriever_tool

# LangGraph components
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
import re
from keywords import KeywordExtractor
from creatingVectorDB import VectorDatabaseManager
import logging

# Load environment variables from .env file
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

# Directory for persisting the vector database
PERSIST_DIRECTORY = "C:\\Users\\hasee\\Desktop\\NCAI\\DomainSpecificChatbotWebAppBackend\\VectorDBs"

finalContext = []

MODEL_NAMES = {
    "domain_check": "qwen2.5:3b",
    "context_selection": "qwen2.5:3b",
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
        rewrite_count: Number of times the query has been rewritten
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rewrite_count: int


class RAGAgent:
    """
    RAG (Retrieval-Augmented Generation) Agent class that handles query processing.
    """
    
    def __init__(self, db_menager, verbose: bool = True, numOfContext=3):
        """
        Initialize the agent and set up its components.

        Args:
            verbose: Whether to print debug information
            numOfContext: Number of documents to retrieve for context
        """
        print("Agent Loading")
        self.verbose = verbose
        self.numOfContext = numOfContext
        self.context = []
        self.context_failure = 0 
        self.embeddingModel = EMBEDDING_MODEL
        self.max_rewrites = 3

        self._build_workflow()
        self.keywordExtractor = KeywordExtractor()  # Placeholder for keyword extractor if needed
        self.vectorDBManager = db_menager
        print("Agent Loaded")


    def _build_workflow(self):
        """Build the LangGraph workflow for query handling."""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("domain_check", self._domain_check)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("context_selection", self._context_selection)
        workflow.add_node("generate", self._generate)
        workflow.add_node("rewrite", self._rewrite)

        # Define edges
        workflow.add_edge(START, "domain_check")

        workflow.add_conditional_edges(
            "domain_check",
            self._route_after_domain_check,
            {
                "retrieve": "retrieve",
                "end": END,
            }
        )

        workflow.add_edge("retrieve", "context_selection")

        workflow.add_conditional_edges(
            "context_selection",
            self._grade_documents,
            {
                "generate": "generate",
                "rewrite": "rewrite",
            }
        )

        workflow.add_edge("generate", END)
        
        workflow.add_conditional_edges(
            "rewrite",
            self._check_rewrite_limit,
            {
                "retrieve": "retrieve",
                "generate": "generate",
            }
        )

        self.graph = workflow.compile()

    def _domain_check(self, state):
        """
        Check if the question belongs to the domain of machine learning.

        Args:
            state: Current state containing the user query

        Returns:
            Updated state with domain check result
        """
        if self.verbose:
            print("---DOMAIN CHECK---")

        class DomainCheck(BaseModel):
            is_ml_domain: str = Field(description="Domain relevance 'yes' or 'no'")

        model = ChatOllama(
            model=MODEL_NAMES["domain_check"],
            temperature=0
        )
        llm_with_tool = model.with_structured_output(DomainCheck)

        prompt = PromptTemplate(
            template="""You are a domain classifier for machine learning questions. \n 
            Here is the user question: {question} \n
            Determine if this question is related to machine learning, artificial intelligence, data science, statistics for ML, or computational learning theory. \n
            Consider topics like: supervised learning, unsupervised learning, reinforcement learning, neural networks, deep learning, 
            statistical learning theory, optimization for ML, feature engineering, model evaluation, etc. \n
            Give a binary score 'yes' or 'no' to indicate whether the question belongs to the machine learning domain.""",
            input_variables=["question"],
        )

        chain = prompt | llm_with_tool

        question = state["messages"][0].content
        scored_result = chain.invoke({"question": question})
        
        if self.verbose:
            print(f"Domain Check Result: {scored_result}")

        # Add domain check result to state
        if scored_result.is_ml_domain == "yes":
            return {"messages": state["messages"], "rewrite_count": 0}
        else:
            # return {"messages": [AIMessage(content="The question is not of the domain of machine learning")], "rewrite_count": 0}
            return {"messages": ["The question is not of the domain of machine learning"], "rewrite_count": 0}

    def _route_after_domain_check(self, state) -> Literal["retrieve", "end"]:
        """Route based on domain check result."""
        last_message = state["messages"][-1]
        if hasattr(last_message, 'content') and last_message.content == "The question is not of the domain of machine learning":
            if self.verbose:
                print("---DECISION: QUESTION NOT IN ML DOMAIN---")
            return "end"
        else:
            if self.verbose:
                print("---DECISION: QUESTION IN ML DOMAIN---")
            return "retrieve"

    def _retrieve(self, state):
        """
        Retrieve documents from the vector database.

        Args:
            state: Current state containing the user query

        Returns:
            Updated state with retrieved documents
        """
        if self.verbose:
            print("---RETRIEVE---")

        query = state["messages"][0].content
        keywords = self.keywordExtractor.extract_keywords(query)

        combined_text = ""
        self.context = []

        #Extract keywords with importance greater than 0.6    
        filtered_keywords = {}
        prompt = query
        for key, value in keywords.items():
            if value > 0.6:
                filtered_keywords[key] = value
                prompt = prompt + " " + key
            print(f"{key}: {value}")
        logging.info(f"Keywords: {', '.join(str(v) for v in filtered_keywords.keys())}" )

        # Hardcode to retrieve 10 contexts
        docs = self.vectorDBManager.search_documents(query=prompt, k=10)
        
        # Store all 10 documents for context selection
        for doc in docs:
            source = os.path.basename(doc.metadata.get('source', 'unknown'))
            page = doc.metadata.get('page', 'unknown')
            self.context.append({"keyword":"","doc": doc, "source": source, "page": page})
        
        combined_text = "\n\n".join(doc.page_content for doc in docs)
        return {"messages": [AIMessage(content=combined_text)], "rewrite_count": state.get("rewrite_count", 0)}

    def _context_selection(self, state):
        """
        Select the top numOfContext documents from the retrieved 10 documents using LLM.

        Args:
            state: Current state containing retrieved documents

        Returns:
            Updated state with selected top contexts
        """
        if self.verbose:
            print("---CONTEXT SELECTION---")

        class ContextSelection(BaseModel):
            selected_indices: List[int] = Field(description="List of indices of the most relevant documents")

        model = ChatOllama(
            model=MODEL_NAMES["context_selection"],
            temperature=0
        )
        llm_with_tool = model.with_structured_output(ContextSelection)

        # Prepare documents with indices for selection
        docs_text = ""
        for i in range(min(10, len(self.context))):
            docs_text += f"Document {i}:\n{self.context[i]['doc'].page_content}\n\n"

        prompt = PromptTemplate(
            template="""You are a document relevance ranker. Given a user question and a list of documents, 
            select the top {num_contexts} most relevant documents that best answer the question.
            
            Question: {question}
            
            Documents:
            {documents}
            
            Return the indices (0-based) of the {num_contexts} most relevant documents as a list of integers.
            For example, if documents 2, 5, and 8 are most relevant, return [2, 5, 8].""",
            input_variables=["question", "documents", "num_contexts"],
        )

        chain = prompt | llm_with_tool

        question = state["messages"][0].content
        
        try:
            scored_result = chain.invoke({
                "question": question, 
                "documents": docs_text,
                "num_contexts": self.numOfContext
            })
            
            selected_indices = scored_result.selected_indices[:self.numOfContext]  # Ensure we don't exceed the limit
            
            if self.verbose:
                print(f"Selected document indices: {selected_indices}")
            
            # Filter context to only selected documents
            selected_context = []
            selected_docs_content = []
            logging.info("Context election")
            logging.info(f"{scored_result=}")
            for i in range(min(10, len(self.context))): 
                logging.info(f"Document {i}:\n{self.context[i]['doc'].page_content}\n\n")
            logging.info(f"{'*'*10}")
            for idx in selected_indices:
                if 0 <= idx < len(self.context):
                    selected_context.append(self.context[idx])
                    selected_docs_content.append(self.context[idx]['doc'].page_content)
            
            # Update self.context to only contain selected documents
            self.context = selected_context
            
            # Combine selected documents content
            combined_text = "\n\n".join(selected_docs_content)
            
        except Exception as e:
            if self.verbose:
                print(f"Context selection failed: {e}. Using first {self.numOfContext} documents.")
                logging.info(f"Context selection failed: {e}. Using first {self.numOfContext} documents.")
            
            # Fallback: use first numOfContext documents
            self.context = self.context[:self.numOfContext]
            combined_text = "\n\n".join(doc['doc'].page_content for doc in self.context)

        return {"messages": [AIMessage(content=combined_text)], "rewrite_count": state.get("rewrite_count", 0)}

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
            if self.verbose:
                print("---DECISION: DOCS RELEVANT---")
            return "generate"
        else:
            if self.verbose:
                print("---DECISION: DOCS NOT RELEVANT---")
            self.context_failure += 1
            if self.context_failure >= 2:
                return "generate"
            return "rewrite"

    def _rewrite(self, state):
        """
        Rewrite the user query to improve retrieval results.

        Args:
            state: Current state containing the original query

        Returns:
            Updated message list with rewritten query and incremented rewrite count
        """
        if self.verbose:
            print("---TRANSFORM QUERY---")

        question = state["messages"][0].content
        current_rewrite_count = state.get("rewrite_count", 0)
        
        msg = [
            HumanMessage(
                content=f""" \n  
        #Look at the input and try to reason about the underlying semantic intent / meaning. \n
        Do not answer the question. Only output an improved question.\n
        Here is the initial question. :
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """
            )
        ]

        model = ChatOllama(
            model=MODEL_NAMES["rewrite"],
            temperature=0
        )
        response = model.invoke(msg)
        response.content = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
        
        if self.verbose:
            print(f"Rewrite count: {current_rewrite_count + 1}")
        
        logging.info("")
        logging.info("Retrieved context was unrelated. Rewriting the question")
        logging.info(f"Rewritten question: {response.content}")

        return {"messages": [response], "rewrite_count": current_rewrite_count + 1}

    def _check_rewrite_limit(self, state) -> Literal["retrieve", "generate"]:
        """
        Check if we've reached the maximum number of rewrites.

        Args:
            state: Current state with rewrite count

        Returns:
            Next action based on rewrite count
        """
        rewrite_count = state.get("rewrite_count", 0)
        
        if rewrite_count >= self.max_rewrites:
            if self.verbose:
                print(f"---DECISION: MAX REWRITES REACHED ({self.max_rewrites})---")
            return "generate"
        else:
            if self.verbose:
                print(f"---DECISION: CONTINUE REWRITING ({rewrite_count}/{self.max_rewrites})---")
            return "retrieve"

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
        
        rewrite_count = state.get("rewrite_count", 0)
        
        if self.context_failure >= 2 or rewrite_count >= self.max_rewrites:
            self.context = []
            # return {"messages": [AIMessage(content="This question is out of my scope.")]}
            return {"messages": ["This question is out of my scope."]}
        

        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOllama(
            model=MODEL_NAMES["generate"],
            temperature=0)

        formatted_prompt = prompt.format(context=docs, question=question)

        if self.verbose:
            print("Final Prompt:\n", formatted_prompt)

        rag_chain = prompt | llm | StrOutputParser()
        response = rag_chain.invoke({"context": docs, "question": question})

        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        
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
            ],
            "rewrite_count": 0
        }

        result = None
        self.context_failure = 0

        for output in self.graph.stream(inputs):
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print(value)
                print("---\n")
                result = value

        extracted_data = []
        for entry in self.context:
            doc = entry.get("doc", {})
            extracted_data.append({
                "book_title": os.path.basename(doc.metadata["source"]).split(".")[0],
                "page_number": doc.metadata["page"],
                "page_content": doc.page_content,
                "keyword" : entry['keyword']
            })
        
        # return {"messages": [AIMessage(content=result["messages"][0])]}, extracted_data

        return result["messages"][0], extracted_data


# Example usage
if __name__ == "__main__":
    rag_agent = RAGAgent(verbose=False)

    # Suppress specific loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.basicConfig(
                    filename="C:\\Users\\hasee\\Desktop\\DomainSpecificChatbotWebAppBackend\\DomainSpecificChatbotWebAppBackend\\V42.log",
                    encoding="utf-8",
                    filemode="w",
                    format="{asctime} - {levelname} - {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M",
                    level=logging.INFO
    )

    questions = [
    "What is the PAC learning model and how does it define learnability?",
    "How is the VC dimension used to analyze the capacity of a hypothesis class?",
    "What is uniform convergence and why is it important in statistical learning theory?",
    "How does the empirical risk minimization (ERM) principle relate to generalization?",
    "What is the bias-variance tradeoff and how does it affect learning performance?",
    "How does the Perceptron algorithm work and what are its convergence guarantees?",
    "What is the definition and role of Rademacher complexity in learning theory?",
    "How do support vector machines (SVMs) find the optimal separating hyperplane?",
    "What is online learning and how does it differ from batch learning?",
    "How do kernel methods enable learning in high-dimensional feature spaces?",
    "what is an ice cream?"
]
    for question in questions:

        # question = "what are cost minimization clusterings?"
        logging.info(f"{'-'*50} {'-'*50}")
        logging.info(f"Question: {question}")
        
        response, context = rag_agent(question)
        logging.info(f"Answer: {response}")
        logging.info("")
        
            
        for part_context in context:
            logging.info(f"Book Title: {part_context['book_title']}")
            logging.info(f"Context: {part_context['page_content']}")
            logging.info("")


    print("\nFinal Response:\n", response)
    print("Final Context:", context)