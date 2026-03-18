import os
from ddgs import DDGS
import trafilatura
from config import SETTINGS

def write_report(filename: str, content: str) -> str:
    """Write content to a markdown report file."""
    os.makedirs(SETTINGS.output_dir, exist_ok=True)
    filepath = os.path.join(SETTINGS.output_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return f"Report saved to {filepath}"


def web_search(query: str) -> list[dict]:
    """Search the web using DuckDuckGo."""
    try:
        results = []
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=SETTINGS.max_search_results):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "")
                })
        return results
    except Exception as e:
        return [{"error": str(e)}]


def read_url(url: str) -> str:
    """Read and extract content from a URL."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return f"Could not fetch URL: {url}"
        
        content = trafilatura.extract(downloaded)
        if content is None:
            return f"Could not extract content from: {url}"
        
        # Limit content length
        if len(content) > SETTINGS.max_url_content_length:
            content = content[:SETTINGS.max_url_content_length] + "...[truncated]"
        
        return content
    except Exception as e:
        return f"Error reading URL {url}: {str(e)}"