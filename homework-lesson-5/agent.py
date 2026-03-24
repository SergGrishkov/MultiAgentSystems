from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from config import SETTINGS, SYSTEM_PROMPT
from retriever import get_retriever
from tools import web_search, read_url, write_report, knowledge_search

llm = ChatOpenAI(
    api_key=SETTINGS.openai_api_key.get_secret_value(),
    model=SETTINGS.model_name,
    temperature=0,
).bind(system=SYSTEM_PROMPT)

@tool
def web_search_tool(query: str) -> list[dict]:
    """Search the web for information using DuckDuckGo."""
    return web_search(query)

@tool
def read_url_tool(url: str) -> str:
    """Read and extract content from a specific URL."""
    return read_url(url)

@tool
def write_report_tool(filename: str, content: str) -> str:
    """Always write a markdown report to a file."""
    return write_report(filename, content)

@tool
def knowledge_search_tool(query: str) -> list[dict]:
    """Search in ingested local knowledge base."""
    return knowledge_search(query)

@tool
def should_use_knowledge_search(query: str) -> bool:
    """Return True when the query is likely to benefit from local knowledge search."""
    # Simple heuristic: use knowledge search if query contains technical terms
    technical_terms = ["RAG", "retrieval", "embedding", "vectorstore", "cross-encoder"]
    return any(term in query for term in technical_terms)

tools = [web_search_tool, read_url_tool, write_report_tool, knowledge_search_tool, should_use_knowledge_search]

memory = MemorySaver()

agent = create_react_agent(
    llm,
    tools,
    checkpointer=memory
)
