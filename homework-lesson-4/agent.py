from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from config import SETTINGS, SYSTEM_PROMPT
from tools import web_search, read_url, write_report

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
    """Write a markdown report to a file."""
    return write_report(filename, content)

tools = [web_search_tool, read_url_tool, write_report_tool]

memory = MemorySaver()

agent = create_react_agent(
    llm,
    tools,
    checkpointer=memory
)
