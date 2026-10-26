from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class LLMHealthResponse(BaseModel):
    """Health diagnostic report for the configured LLM provider."""

    provider: str = Field(..., description="Selected LLM provider ('ollama', 'groq', 'mock')")
    model: str = Field(..., description="Configured model identifier")
    is_reachable: bool = Field(..., description="Whether the provider endpoint is reachable")
    is_model_installed: bool = Field(default=True, description="Whether configured model is installed (Ollama)")
    available_models: List[str] = Field(default_factory=list, description="Installed models discovered at provider endpoint")
    generation_successful: bool = Field(default=False, description="Whether test generation succeeded")
    test_latency_ms: Optional[float] = Field(default=None, description="Test generation latency in milliseconds")
    error: Optional[str] = Field(default=None, description="Descriptive error if health check failed")


class HealthResponse(BaseModel):
    """Health check response payload."""

    status: str = Field(default="healthy", description="Status of the API service")
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application semantic version")
    environment: str = Field(..., description="Current running environment")
    timestamp: datetime = Field(..., description="Timestamp of the health check in UTC")
    llm: Optional[LLMHealthResponse] = Field(default=None, description="LLM provider health diagnostic details")

