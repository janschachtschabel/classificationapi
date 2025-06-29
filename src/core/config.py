"""Application configuration settings."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_default_model: str = Field(default="gpt-4.1-mini", description="Default OpenAI model to use")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_debug: bool = Field(default=False, description="Enable debug mode")
    api_reload: bool = Field(default=False, description="Enable auto-reload")

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(default="json", description="Log format")

    # Cache Configuration
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache size")

    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(default=60, description="Rate limit requests per minute")

    # Request Timeouts
    http_timeout_seconds: int = Field(default=30, description="HTTP request timeout in seconds")
    openai_timeout_seconds: int = Field(default=60, description="OpenAI API timeout in seconds")

    # Default LLM Settings
    default_temperature: float = Field(default=0.2, description="Default temperature for LLM requests")
    default_max_tokens: int = Field(default=15000, description="Default max tokens for LLM requests")

    # Large Vocabulary Processing Settings
    large_vocab_threshold: int = Field(
        default=200, description="Threshold for considering a vocabulary 'large' and using block processing"
    )
    vocab_block_size: int = Field(
        default=200, description="Number of concepts per block for large vocabulary processing"
    )
    max_candidates_per_block: int = Field(default=5, description="Maximum candidates to keep from each block")
    final_refinement_limit: int = Field(
        default=5, description="Always perform final refinement with top candidates from all blocks"
    )

    # SKOS Parsing Limits
    max_concepts_standard: int = Field(
        default=200, description="Maximum concepts to extract from standard vocabularies"
    )
    max_concepts_large: int = Field(
        default=5000, description="Maximum concepts to extract from large vocabularies (when using block processing)"
    )
    max_depth_standard: int = Field(default=3, description="Maximum recursion depth for standard vocabularies")
    max_depth_large: int = Field(default=5, description="Maximum recursion depth for large vocabularies")
    max_narrower_concepts: int = Field(default=30, description="Maximum narrower concepts to process per concept")

    # Supported Models
    supported_models: list[str] = Field(
        default=["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"],
        description="List of supported OpenAI models",
    )


# Global settings instance
try:
    settings = Settings()  # type: ignore[call-arg]
except Exception:
    # Fallback for testing or when environment variables are not set
    # Provide all required fields with test values
    settings = Settings(openai_api_key="test-key")
