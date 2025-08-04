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
PERSIST_DIRECTORY = "C:\\Users\\hasee\\Desktop\\DomainSpecificChatbotWebAppBackend\\DomainSpecificChatbotWebAppBackend\\VectorDBs"
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
        print("Agent Loading")
        self.verbose = verbose
        self.numOfContext = numOfContext
        self.context = []
        self.context_failure = 0 
        self.embeddingModel = EMBEDDING_MODEL

        self._build_workflow()
        self.keywordExtractor = KeywordExtractor()  # Placeholder for keyword extractor if needed
        self.vectorDBManager = VectorDatabaseManager(documents_directory=PERSIST_DIRECTORY, model_name=EMBEDDING_MODEL, collection_name="rag-chroma")
        print("Agent Loaded")


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

        combined_text = ""
        self.context = []

        #Extract keywords with importance greater than 0.7    
        filtered_keywords = {}
        prompt = query
        for key, value in keywords.items():
            if value >0.6:
                filtered_keywords[key] = value
                prompt = prompt + " " + key
            print(f"{key}: {value}")
        logging.info(f"Keywords: {', '.join(str(v) for v in filtered_keywords.keys())}" )

        
        docs = self.vectorDBManager.search_documents(query=prompt, k=self.numOfContext)
        combined_text = ""
        for doc in docs:
            source = os.path.basename(doc.metadata.get('source', 'unknown'))
            page = doc.metadata.get('page', 'unknown')
            self.context.append({"keyword":"","doc": doc, "source": source, "page": page})
        
        combined_text += "\n\n".join(doc.page_content for doc in docs)
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
        logging.info("Retrieved context was unrelated. Rewriting the question")
        logging.info(f"Rewritten question: {response.content}")

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
        

        return result["messages"][0], extracted_data


# Example usage
if __name__ == "__main__":
    rag_agent = RAGAgent(verbose=False)

    # Suppress specific loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.basicConfig(
                    filename="C:\\Users\\hasee\\Desktop\\DomainSpecificChatbotWebAppBackend\\DomainSpecificChatbotWebAppBackend\\V3.log",
                    encoding="utf-8",
                    filemode="w",
                    format="{asctime} - {levelname} - {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M",
                    level=logging.INFO
    )

    questions = questions = [
    "What is the PAC learning model and how does it define learnability?",
    "How is the VC dimension used to analyze the capacity of a hypothesis class?",
    "What is uniform convergence and why is it important in statistical learning theory?",
    "How does the empirical risk minimization (ERM) principle relate to generalization?",
    "What is the bias-variance tradeoff and how does it affect learning performance?",
    "How does the Perceptron algorithm work and what are its convergence guarantees?",
    "What is the definition and role of Rademacher complexity in learning theory?",
    "How do support vector machines (SVMs) find the optimal separating hyperplane?",
    "What is online learning and how does it differ from batch learning?",
    "How do kernel methods enable learning in high-dimensional feature spaces?"
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