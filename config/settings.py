"""
Configuration management using Pydantic and environment variables.
"""
import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",  # Allow extra fields like groq_api_key
    )
    
    # Celonis Configuration
    celonis_url: str = Field(..., description="Celonis team URL")
    celonis_api_token: str = Field(..., description="Celonis API token")
    celonis_key_type: str = Field(default="USER_KEY", description="USER_KEY or APP_KEY")

    # Add to your Settings class
    data_model_id: Optional[str] = Field(None, description="Data Model ID")
    data_pool_name: Optional[str] = Field(None, description="Data Pool Name")
    data_pool_id: Optional[str] = Field(None, description="Data Pool ID")
    
    # Groq LLM Configuration (legacy)
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key for LLM translation")

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key for PQL translation")
    
    # Analysis Configuration
    analysis_id: Optional[str] = Field(None, description="Studio View/Analysis ID")
    knowledge_model_id: Optional[str] = Field(None, description="Knowledge Model ID")

    # Studio/Package configuration for KM/View extraction
    space_id: Optional[str] = Field(None, description="Celonis Studio Space ID")
    package_id: Optional[str] = Field(None, description="Celonis Studio Package ID")
    
    # Output Configuration
    output_dir: Path = Field(default=Path("./data/raw"), description="Output directory")
    log_level: str = Field(default="INFO", description="Logging level")
    log_dir: Path = Field(default=Path("./logs"), description="Log directory")

    # Power BI Configuration
    pbi_workspace_id: Optional[str] = Field(default=None, description="Power BI Workspace ID")
    pbi_dataset_id: Optional[str] = Field(default=None, description="Power BI Dataset ID")
    
    @field_validator("celonis_url")
    @classmethod
    def validate_celonis_url(cls, v: str) -> str:
        """Ensure URL starts with https:// and strip whitespace."""
        v = v.strip()
        if not v.startswith("https://"):
            v = f"https://{v}"
        if not (".celonis.cloud" in v or ".celonis.eu" in v):
            raise ValueError("URL must be a valid Celonis cloud URL")
        return v
    
    @field_validator("output_dir", "log_dir")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Create directories if they don't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


def get_settings() -> Settings:
    """
    Factory function to get settings instance.
    Validates that required environment variables are set.
    """
    settings = Settings()
    
    # Additional validation
    if not settings.celonis_api_token or settings.celonis_api_token == "your-api-token-here":
        raise ValueError(
            "CELONIS_API_TOKEN is not set. "
            "Get it from Celonis Studio → Profile → Edit Profile → Create API Key"
        )
    
    return settings