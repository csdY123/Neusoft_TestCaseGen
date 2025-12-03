# @Author：klh
from openai import OpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage


def create_vllm_client(base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
    """Create a vLLM OpenAI-compatible client"""
    return OpenAI(api_key=api_key, base_url=base_url)


def generate_response_vllm(client: OpenAI, model_id: str, prompt: str, system_prompt: str = "",
                           temperature: float = 0.7, max_tokens: int = 4096):
    """Generate response using vLLM (non-streaming)"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    return response.choices[0].message.content


def generate_response_vllm_stream(client: OpenAI, model_id: str, prompt: str, system_prompt: str = "",
                                  temperature: float = 0.7, max_tokens: int = 4096):
    """Generate response using vLLM (streaming) - yields partial content"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    content = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            chunk_content = chunk.choices[0].delta.content
            content += chunk_content
            yield content


def generate_response_ollama(llm: BaseChatModel, prompt: str, system_prompt: str = "", enable_thinking: bool = False):
    """Generate response using Ollama (LangChain)"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]
    ai_message = llm.invoke(messages)
    return ai_message.content if hasattr(ai_message, 'content') else str(ai_message)


def generate_response(llm, prompt, system_prompt="", enable_thinking=False):
    """Legacy function for backward compatibility with OllamaLLM"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]
    ai_message = llm.invoke(messages)
    print(ai_message)
    return ai_message
