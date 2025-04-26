import ollama
from typing import Any, Dict, List, Mapping, Optional, Union, Iterator
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
    Callbacks
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import re
from langchain_core.outputs import ChatGeneration, ChatResult

class OllamaChat(BaseChatModel):
    """Chat model implementation for Ollama API using ollama Python package."""
    
    model_name: str = "deepseek-r1:1.5b"
    temperature: float = 0.0
    streaming: bool = False
    
    def _convert_to_ollama_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to Ollama message format."""
        ollama_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                role = "user"  # Default fallback
                
            ollama_messages.append({
                "role": role,
                "content": message.content
            })
        return ollama_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response using the Ollama API."""
        # Convert LangChain messages to Ollama format
        ollama_messages = self._convert_to_ollama_messages(messages)
        
        # Prepare parameters
        params = {
            "model": self.model_name,
            "messages": ollama_messages,
        }
        
        # Add stop sequences if provided
        if stop:
            params["options"] = {"stop": stop}
            
        # Add any additional parameters
        for key, value in kwargs.items():
            if key == "options" and isinstance(value, dict):
                params.setdefault("options", {}).update(value)
            else:
                params[key] = value
            
        # Make the API request using ollama package
        response = ollama.chat(**params)
        response["message"]["content"] = re.sub(r"<think>.*?</think>", "", response["message"]["content"], flags=re.DOTALL).strip() 
        
        # Create ChatGeneration object
        generation = ChatGeneration(
            message=AIMessage(content=response["message"]["content"]),
            generation_info={"model": self.model_name, 
                           "finish_reason": response.get("done", True) and "stop" or "unknown"}
        )
        
        # Return the ChatResult
        return ChatResult(generations=[generation])
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """Stream the chat response from Ollama API."""
        if not self.streaming:
            yield next(iter(self._generate(messages, stop, run_manager, **kwargs).generations))
            return
            
        # Convert LangChain messages to Ollama format
        ollama_messages = self._convert_to_ollama_messages(messages)
        
        # Prepare parameters
        params = {
            "model": self.model_name,
            "messages": ollama_messages,
            "temperature": self.temperature,
            "stream": True
        }
        
        # Add stop sequences if provided
        if stop:
            params["options"] = {"stop": stop}
            
        # Add any additional parameters
        for key, value in kwargs.items():
            if key == "options" and isinstance(value, dict):
                params.setdefault("options", {}).update(value)
            else:
                params[key] = value
        
        # Stream the response
        content = ""
        for chunk in ollama.chat(**params):
            if "message" in chunk:
                content_chunk = chunk["message"].get("content", "")
                content += content_chunk
                
                # Create a generation with the accumulated content
                generation = ChatGeneration(
                    message=AIMessage(content=content),
                    generation_info={"model": self.model_name}
                )
                
                # Update callback manager with new token
                if run_manager and content_chunk:
                    run_manager.on_llm_new_token(content_chunk)
                
                yield generation
    
    def invoke(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Invoke the chat model with the given messages.

        Args:
            messages: The messages to generate a response for
            stop: A list of strings to stop generation when encountered
            callbacks: Callbacks to pass to the LLM
            **kwargs: Additional parameters to pass to the underlying model

        Returns:
            The generated AIMessage
        """
        result = self.generate(
            [messages], stop=stop, callbacks=callbacks, **kwargs
        )
        return result.generations[0][0].message
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "ollama-chat"