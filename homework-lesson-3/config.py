from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: SecretStr
    model_name: str = "gpt-4o-mini"

    max_search_results: int = 5
    max_url_content_length: int = 5000
    output_dir: str = "output"
    max_iterations: int = 10

    model_config = {"env_file": ".env"}


SETTINGS = Settings()

SYSTEM_PROMPT = """You are a helpful research assistant. When asked a question.
Always provide structured, well-organized reports in Markdown format."""
