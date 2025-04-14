"""
RAG Retrieval Agent Application

This module implements a Retrieval Augmented Generation (RAG) system using LangChain and LangGraph.
The system can:
1. Load and access a vector database stored locally
2. Process user queries through an agent that decides when to retrieve information
3. Grade document relevance and rewrite queries when necessary
4. Generate responses based on retrieved documents

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

# Suppress warnings
warnings.filterwarnings("ignore")

# Import required libraries
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# LangChain imports
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

# LangGraph imports
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the persistent directory for vector database
PERSIST_DIRECTORY = "chroma_db" 
finalContext = []

class AgentState(TypedDict):
    """
    Type definition for the agent state.
    
    Attributes:
        messages: Sequence of messages in the conversation
    """
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

class LoggingRetriever(BaseRetriever):
    """
    Custom retriever that logs the documents retrieved and filters duplicates.
    
    Attributes:
        base_retriever: The underlying retriever to use
        seen_hashes: Set of document hashes that have already been seen
    """
    base_retriever: BaseRetriever = Field(...)
    seen_hashes: set = Field(default_factory=set)
    
    def _get_relevant_documents(self, query, *, run_manager=None):
        """
        Retrieves relevant documents and logs them.
        
        Args:
            query: The query to search for
            run_manager: Optional run manager for tracking
            
        Returns:
            List of unique documents relevant to the query
        """
        docs = self.base_retriever._get_relevant_documents(query, run_manager=run_manager)
        unique_docs = []
        docs_with_metadata = []
        for doc in docs:
            # Create a unique hash of content + metadata
            doc_hash = hash(f"{doc.page_content}-{doc.metadata}")
            if doc_hash not in self.seen_hashes:
                self.seen_hashes.add(doc_hash)
                unique_docs.append(doc)
                source = os.path.basename(doc.metadata.get('source', 'unknown'))
                page = doc.metadata.get('page', 'unknown')
                docs_with_metadata.append({"doc": doc, "source": source, "page": page})
                print(f"Retrieved: {source} p.{page} - {doc.page_content[:50]}...")
        global finalContext 
        finalContext = docs_with_metadata
        
        return unique_docs

class RAGAgent:
    """
    RAG (Retrieval Augmented Generation) Agent that processes user queries
    by retrieving relevant documents and generating responses.
    """
    
    def __init__(self, verbose: bool = True, numOfContext=3):
        """
        Initialize the RAG agent.
        
        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        self.client = OpenAI()
        
        # Load the vector database
        self._load_vector_db()
        
        # Set up the retriever and tools
        self.numOfContext = numOfContext
        self._setup_retriever()
        
        # Build the agent workflow
        self._build_workflow()
        self.context = []

    def _load_vector_db(self):
        """Load the vector database from the persistent directory."""
        if self.verbose:
            print(f"Loading vector database from {PERSIST_DIRECTORY}")
        
        embedding_function = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name="rag-chroma",
            embedding_function=embedding_function,
            persist_directory=PERSIST_DIRECTORY
        )
        
        if self.verbose:
            print(f"Vector database loaded successfully")
    
    def _setup_retriever(self):
        """Set up the retriever with logging and create the retriever tool."""
        # Initialize the retriever with MMR search
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={
                "k": self.numOfContext,  # Fetch more documents initially
                "fetch_k": 20,  # Larger candidate pool
                "lambda_mult": 0.5  # Diversity control
            }
        )
        
        # Wrap with logging retriever
        self.logging_retriever = LoggingRetriever(base_retriever=retriever)
        
        # Create retriever tool
        self.retriever_tool = create_retriever_tool(
            self.logging_retriever,
            "retrieve_relevant_section",
            "Search and return information from the documents"
        )
        
        self.tools = [self.retriever_tool]
    
    def _build_workflow(self):
        """Build the workflow graph for the agent."""
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Define the nodes
        workflow.add_node("agent", self._agent)  # Agent node
        retrieve = ToolNode([self.retriever_tool])
        workflow.add_node("retrieve", retrieve)  # Retrieval node
        workflow.add_node("rewrite", self._rewrite)  # Query rewriting node
        workflow.add_node("generate", self._generate)  # Response generation node
        
        # Define the edges
        workflow.add_edge(START, "agent")
        
        # Conditional edge from agent node
        workflow.add_conditional_edges(
            "agent",
            tools_condition,  # Assess agent decision
            {
                "tools": "retrieve",  # If tools are needed, go to retrieve
                END: END,  # Otherwise, end
            },
        )
        
        # Conditional edge from retrieve node
        workflow.add_conditional_edges(
            "retrieve",
            self._grade_documents,  # Grade document relevance
            {
                "generate": "generate",  # If documents are relevant, generate response
                "rewrite": "rewrite",  # If not, rewrite the query
            }
        )
        
        # Final edges
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")
        
        # Compile the graph
        self.graph = workflow.compile()
    
    def _agent(self, state):
        """
        Agent node: decides whether to retrieve or directly answer.
        
        Args:
            state: Current state with messages
            
        Returns:
            Updated state with agent response
        """
        if self.verbose:
            print("---CALL AGENT---")
        
        messages = state["messages"]
        model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
        model = model.bind_tools(self.tools)
        response = model.invoke(messages)
        
        return {"messages": [response]}
    
    def _grade_documents(self, state) -> Literal["generate", "rewrite"]:
        """
        Grades document relevance to determine next action.
        
        Args:
            state: Current state with messages and retrieved documents
            
        Returns:
            Decision string ("generate" or "rewrite")
        """
        if self.verbose:
            print("---CHECK RELEVANCE---")
        
        # Data model for grading
        class Grade(BaseModel):
            """Binary score for relevance check."""
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")
        
        # LLM setup
        model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
        llm_with_tool = model.with_structured_output(Grade)
        
        # Prompt template
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )
        
        # Create chain
        chain = prompt | llm_with_tool
        
        # Extract information from state
        messages = state["messages"]
        last_message = messages[-1]
        question = messages[0].content
        docs = last_message.content
        
        # Grade documents
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score

        global finalContext
        self.context = finalContext

        if score == "yes":
            if self.verbose:
                print("---DECISION: DOCS RELEVANT---")
            self.context = finalContext
            return "generate"
        else:
            if self.verbose:
                print("---DECISION: DOCS NOT RELEVANT---")
            return "rewrite"
    
    def _rewrite(self, state):
        """
        Rewrites the query to improve retrieval results.
        
        Args:
            state: Current state with messages
            
        Returns:
            Updated state with rewritten query
        """
        if self.verbose:
            print("---TRANSFORM QUERY---")
        
        messages = state["messages"]
        question = messages[0].content
        
        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
            )
        ]
        
        # Use LLM to rewrite query
        model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
        response = model.invoke(msg)
        
        return {"messages": [response]}
    
    def _generate(self, state):
        """
        Generates a response based on retrieved documents.
        
        Args:
            state: Current state with messages and relevant documents
            
        Returns:
            Updated state with generated response
        """
        if self.verbose:
            print("---GENERATE---")
        
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]
        docs = last_message.content
        
        # Get RAG prompt from hub
        prompt = hub.pull("rlm/rag-prompt")
        
        # Set up LLM
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
        
        # Format the prompt
        formatted_prompt = prompt.format(context=docs, question=question)
        
        if self.verbose:
            print("Final Prompt:\n", formatted_prompt)
        
        # Create and run chain
        rag_chain = prompt | llm | StrOutputParser()
        response = rag_chain.invoke({"context": docs, "question": question})
        
        return {"messages": [response]}
    
    def __call__(self, query: str) -> str:
        """
        Process a user query through the RAG workflow.
        
        Args:
            query: User query string
            
        Returns:
            Generated response
        """
        # Prepare the input
        inputs = {
            "messages": [
                HumanMessage(content=query)
            ]
        }
        
        # Execute the graph
        result = None
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print(value)
                print("---")
                # print(value, indent=2, width=80, depth=None)
                print("\n---\n")
                result = value
        return result["messages"][0].content, self.context
        # return "Failed to generate a response."

# Example usage
if __name__ == "__main__":
    # Initialize RAG agent
    rag_agent = RAGAgent(verbose=True)
    
    # Process a query
    response, context = rag_agent("What is strong AI?")
    print("\nFinal Response:\n", response)
    print("final COntext:", context)