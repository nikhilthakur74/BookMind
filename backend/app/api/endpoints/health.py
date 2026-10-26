"""Health check and provider diagnostics endpoint implementation."""

import asyncio
from datetime import datetime, timezone
import logging
import time
from typing import Optional
from fastapi import APIRouter, Depends
import httpx

from app.core.config import Settings, get_settings
from app.schemas.health import HealthResponse, LLMHealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


async def check_llm_health(
    run_generation_test: bool = False,
    settings: Optional[Settings] = None,
) -> LLMHealthResponse:
    """Diagnose reachability, model installation, and generation for the configured LLM provider."""
    from app.core.config import get_settings as _get_settings

    cfg = settings or _get_settings()
    provider = cfg.LLM_PROVIDER.lower().strip()

    if provider == "ollama":
        model = cfg.LLM_MODEL or cfg.OLLAMA_MODEL
        base_url = cfg.OLLAMA_BASE_URL.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                tags_resp = await client.get(f"{base_url}/api/tags")

            if tags_resp.status_code != 200:
                return LLMHealthResponse(
                    provider=provider,
                    model=model,
                    is_reachable=False,
                    is_model_installed=False,
                    available_models=[],
                    generation_successful=False,
                    error=f"Ollama returned HTTP {tags_resp.status_code} on /api/tags: {tags_resp.text}",
                )

            tags_data = tags_resp.json()
            available = [m.get("name", "") for m in tags_data.get("models", [])]
            installed = False
            for m in available:
                if (
                    m == model
                    or m == f"{model}:latest"
                    or (":" not in model and m.split(":")[0] == model)
                    or (":" in model and m.startswith(model))
                ):
                    installed = True
                    break

            if not installed:
                return LLMHealthResponse(
                    provider=provider,
                    model=model,
                    is_reachable=True,
                    is_model_installed=False,
                    available_models=available,
                    generation_successful=False,
                    error=(
                        f"Model '{model}' is not installed in Ollama. "
                        f"Available: {available or 'None'}. Run 'ollama pull {model}'."
                    ),
                )

            # Optional test generation probe
            gen_success = False
            latency_ms = None
            gen_error = None

            if run_generation_test:
                t0 = time.perf_counter()
                try:
                    async with httpx.AsyncClient(timeout=cfg.OLLAMA_TIMEOUT_SECONDS) as client:
                        gen_resp = await client.post(
                            f"{base_url}/api/chat",
                            json={
                                "model": model,
                                "messages": [{"role": "user", "content": "Respond with 'ready'"}],
                                "options": {
                                    "temperature": 0.0,
                                    "num_predict": 10,
                                },
                                "stream": False,
                            },
                        )
                    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                    if gen_resp.status_code == 200:
                        gen_success = True
                    else:
                        gen_error = f"Test generation returned HTTP {gen_resp.status_code}: {gen_resp.text}"
                except Exception as e:
                    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                    gen_error = f"Test generation failed: {e}"

            return LLMHealthResponse(
                provider=provider,
                model=model,
                is_reachable=True,
                is_model_installed=True,
                available_models=available,
                generation_successful=gen_success,
                test_latency_ms=latency_ms,
                error=gen_error,
            )

        except (httpx.ConnectError, httpx.NetworkError):
            return LLMHealthResponse(
                provider=provider,
                model=model,
                is_reachable=False,
                is_model_installed=False,
                available_models=[],
                generation_successful=False,
                error=f"Ollama is unreachable at {base_url}. Ensure Ollama is running ('ollama serve').",
            )
        except Exception as e:
            return LLMHealthResponse(
                provider=provider,
                model=model,
                is_reachable=False,
                is_model_installed=False,
                available_models=[],
                generation_successful=False,
                error=f"Failed checking Ollama health: {e}",
            )

    elif provider == "groq":
        model = cfg.LLM_MODEL or cfg.GROQ_MODEL
        if not cfg.GROQ_API_KEY:
            return LLMHealthResponse(
                provider=provider,
                model=model,
                is_reachable=False,
                is_model_installed=False,
                available_models=[],
                generation_successful=False,
                error="GROQ_API_KEY is not configured in backend/.env",
            )

        gen_success = False
        latency_ms = None
        gen_error = None

        if run_generation_test:
            t0 = time.perf_counter()
            try:
                from groq import AsyncGroq

                groq_client = AsyncGroq(api_key=cfg.GROQ_API_KEY, timeout=cfg.LLM_TIMEOUT_SECONDS)
                resp = await groq_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5,
                )
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                gen_success = bool(resp.choices and resp.choices[0].message.content)
            except Exception as e:
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                gen_error = f"Groq test generation failed: {e}"

        return LLMHealthResponse(
            provider=provider,
            model=model,
            is_reachable=True,
            is_model_installed=True,
            available_models=[model],
            generation_successful=gen_success,
            test_latency_ms=latency_ms,
            error=gen_error,
        )

    elif provider == "mock":
        return LLMHealthResponse(
            provider=provider,
            model=cfg.LLM_MODEL or "mock-llama-3.3-70b",
            is_reachable=True,
            is_model_installed=True,
            available_models=["mock-llama-3.3-70b"],
            generation_successful=True,
            test_latency_ms=1.0,
            error=None,
        )

    return LLMHealthResponse(
        provider=provider,
        model=cfg.LLM_MODEL or "unknown",
        is_reachable=False,
        is_model_installed=False,
        available_models=[],
        generation_successful=False,
        error=f"Unknown LLM provider: '{provider}'",
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns service availability, app name, version, running environment, and LLM diagnostics.",
)
async def get_health(
    probe_llm: bool = False,
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Return application health information and LLM provider connectivity status."""
    llm_info = None
    try:
        llm_info = await check_llm_health(run_generation_test=probe_llm, settings=settings)
    except Exception as e:
        logger.warning(f"Error gathering LLM health info: {e}")

    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(timezone.utc),
        llm=llm_info,
    )


@router.get(
    "/health/llm",
    response_model=LLMHealthResponse,
    summary="LLM Provider Health Diagnostics",
    description="Performs in-depth provider diagnostics including reachability, model verification, and a test generation prompt.",
)
async def get_llm_health(
    settings: Settings = Depends(get_settings),
) -> LLMHealthResponse:
    """Return in-depth LLM diagnostics with active test generation."""
    return await check_llm_health(run_generation_test=True, settings=settings)

