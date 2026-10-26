"""LLM client integration for BookMind Phase 4 synthesis.

Supports:
- Groq ultra-fast inference (Llama 3.3 70B Versatile) with JSON mode
- Structured Pydantic response validation
- Automatic retry on malformed JSON or validation errors
- Deterministic Mock LLM client for isolated unit and integration testing
"""

import abc
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(abc.ABC):
    """Abstract interface for LLM synthesis."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Configured model name."""
        pass

    @property
    @abc.abstractmethod
    def prompt_version(self) -> str:
        """Prompt template version identifier."""
        pass

    @abc.abstractmethod
    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        max_retries: Optional[int] = None,
    ) -> T:
        """Generate and validate a structured response matching the Pydantic schema."""
        pass


class OllamaLLMClient(BaseLLMClient):
    """Local Ollama LLM client supporting schema-constrained JSON generation.

    Features:
    - Bounded context window and configurable token generation limits.
    - Automatic validation of model installation via Ollama /api/tags.
    - Fast, native constrained decoding using Ollama JSON schemas.
    - Automatic retry with error feedback on validation or decode failures.
    - Graceful error reporting when Ollama server is offline or model is missing.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        keep_alive: Optional[str] = None,
        max_retries: Optional[int] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        raw_url = base_url or settings.OLLAMA_BASE_URL
        self._base_url = raw_url.rstrip("/")
        self._model_name = model_name or settings.LLM_MODEL or settings.OLLAMA_MODEL
        self._prompt_version = prompt_version or settings.PROMPT_VERSION
        self._temperature = temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
        self._timeout = timeout if timeout is not None else settings.OLLAMA_TIMEOUT_SECONDS
        self._num_ctx = num_ctx if num_ctx is not None else settings.OLLAMA_NUM_CTX
        self._num_predict = num_predict if num_predict is not None else settings.OLLAMA_NUM_PREDICT
        self._keep_alive = keep_alive if keep_alive is not None else settings.OLLAMA_KEEP_ALIVE
        self._max_retries = max_retries if max_retries is not None else settings.LLM_MAX_RETRIES
        self._custom_client = http_client
        self._client: Optional[httpx.AsyncClient] = None
        self._model_verified = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    @property
    def base_url(self) -> str:
        return self._base_url

    def _get_client(self) -> httpx.AsyncClient:
        if self._custom_client is not None:
            return self._custom_client
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=5.0),
            )
        return self._client

    async def aclose(self) -> None:
        """Close underlying HTTP client if managed internally."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def check_connection_and_model(self) -> Dict[str, Any]:
        """Check if Ollama server is reachable and configured model is installed."""
        client = self._get_client()
        try:
            resp = await client.get(f"{self._base_url}/api/tags", timeout=5.0)
            if resp.status_code != 200:
                return {
                    "reachable": False,
                    "installed": False,
                    "models": [],
                    "error": f"Ollama returned HTTP {resp.status_code} on /api/tags: {resp.text}",
                }
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            installed = False
            for m in models:
                if (
                    m == self._model_name
                    or m == f"{self._model_name}:latest"
                    or (":" not in self._model_name and m.split(":")[0] == self._model_name)
                    or (":" in self._model_name and m.startswith(self._model_name))
                ):
                    installed = True
                    break

            if not installed:
                return {
                    "reachable": True,
                    "installed": False,
                    "models": models,
                    "error": (
                        f"Ollama model '{self._model_name}' is not installed at {self._base_url}. "
                        f"Installed models: {models or 'None'}. Run 'ollama pull {self._model_name}' to install it."
                    ),
                }

            return {
                "reachable": True,
                "installed": True,
                "models": models,
                "error": None,
            }
        except (httpx.ConnectError, httpx.NetworkError):
            return {
                "reachable": False,
                "installed": False,
                "models": [],
                "error": f"Ollama service is unreachable at {self._base_url}. Ensure Ollama is running ('ollama serve').",
            }
        except httpx.TimeoutException:
            return {
                "reachable": False,
                "installed": False,
                "models": [],
                "error": f"Connection to Ollama service at {self._base_url} timed out.",
            }
        except Exception as e:
            return {
                "reachable": False,
                "installed": False,
                "models": [],
                "error": f"Failed checking Ollama status: {e}",
            }

    async def validate_model_installed(self) -> None:
        """Validate connection and model installation, raising RuntimeError with action guidance on failure."""
        status = await self.check_connection_and_model()
        if not status["reachable"] or not status["installed"]:
            raise RuntimeError(status["error"] or f"Ollama validation failed for model '{self._model_name}'")
        self._model_verified = True

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Strip markdown code fence blocks or surrounding commentary from model output."""
        s = text.strip()
        if s.startswith("```json"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
        first_brace = s.find("{")
        last_brace = s.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return s[first_brace : last_brace + 1]
        return s

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        max_retries: Optional[int] = None,
    ) -> T:
        if not self._model_verified:
            await self.validate_model_installed()

        retries = max_retries if max_retries is not None else self._max_retries
        schema = response_model.model_json_schema()
        schema_json = json.dumps(schema, indent=2)

        augmented_system_prompt = (
            f"{system_prompt}\n\n"
            f"CRITICAL: You MUST respond with ONLY a valid JSON object matching the following JSON Schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Do not include any preamble, commentary, or markdown formatting outside the JSON."
        )

        messages = [
            {"role": "system", "content": augmented_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        client = self._get_client()
        last_error = None

        for attempt in range(1, retries + 1):
            try:
                payload: Dict[str, Any] = {
                    "model": self._model_name,
                    "messages": messages,
                    "format": schema,
                    "options": {
                        "temperature": self._temperature,
                        "num_ctx": self._num_ctx,
                        "num_predict": self._num_predict,
                    },
                    "keep_alive": self._keep_alive,
                    "stream": False,
                }

                response = await client.post(
                    f"{self._base_url}/api/chat", json=payload, timeout=self._timeout
                )

                if response.status_code != 200:
                    # If format: schema fails with 400 on older versions, fall back to format: "json"
                    if response.status_code == 400 and payload.get("format") == schema:
                        logger.warning("Ollama rejected schema format; falling back to format='json'")
                        payload["format"] = "json"
                        response = await client.post(
                            f"{self._base_url}/api/chat", json=payload, timeout=self._timeout
                        )

                    if response.status_code != 200:
                        raise RuntimeError(
                            f"Ollama API returned HTTP {response.status_code}: {response.text}"
                        )

                resp_data = response.json()
                msg = resp_data.get("message", {})
                raw_text = msg.get("content", "").strip()

                if not raw_text:
                    raise json.JSONDecodeError("Ollama returned empty response content", doc="", pos=0)

                clean_json = self._clean_json_text(raw_text)
                parsed_dict = json.loads(clean_json)
                return response_model.model_validate(parsed_dict)

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(
                    f"Ollama structured generation attempt {attempt}/{retries} failed validation: {e}. Retrying..."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response produced a validation error: {str(e)}. "
                            f"Please rectify the JSON according to the schema and return ONLY valid JSON."
                        ),
                    }
                )
                await asyncio.sleep(0.5 * attempt)

            except httpx.ConnectError as e:
                raise RuntimeError(
                    f"Ollama service is unreachable at {self._base_url}. Ensure Ollama is running ('ollama serve')."
                ) from e

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Ollama request timed out after {self._timeout}s on attempt {attempt}/{retries}: {e}")
                if attempt == retries:
                    raise TimeoutError(
                        f"Ollama generation timed out after {self._timeout}s while generating {response_model.__name__}."
                    ) from e
                await asyncio.sleep(1.0 * attempt)

            except Exception as e:
                last_error = e
                logger.error(f"Ollama generation error on attempt {attempt}/{retries}: {e}")
                if attempt == retries:
                    raise
                await asyncio.sleep(1.0 * attempt)

        raise RuntimeError(
            f"Failed to generate valid structured response for {response_model.__name__} "
            f"after {retries} attempts: {last_error}"
        )


class GroqLLMClient(BaseLLMClient):
    """Production Groq LLM client using llama-3.3-70b-versatile with JSON mode."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ):
        self._api_key = api_key if api_key is not None else settings.GROQ_API_KEY
        self._model_name = model_name or settings.GROQ_MODEL
        self._prompt_version = prompt_version or settings.PROMPT_VERSION
        self._temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self._timeout = timeout if timeout is not None else settings.LLM_TIMEOUT_SECONDS
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "GROQ_API_KEY is not configured. Set GROQ_API_KEY in backend/.env "
                    "or use MockLLMClient for testing."
                )
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(api_key=self._api_key, timeout=self._timeout)
            except ImportError as e:
                raise RuntimeError(f"groq library is required for GroqLLMClient: {e}") from e

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        max_retries: Optional[int] = None,
    ) -> T:
        self._ensure_client()
        retries = max_retries if max_retries is not None else settings.LLM_MAX_RETRIES

        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        augmented_system_prompt = (
            f"{system_prompt}\n\n"
            f"CRITICAL: You MUST respond with ONLY a valid JSON object matching the following JSON Schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Do not include any preamble, commentary, or markdown formatting outside the JSON."
        )

        messages = [
            {"role": "system", "content": augmented_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error = None
        for attempt in range(1, retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                )

                raw_text = response.choices[0].message.content or "{}"
                data = json.loads(raw_text)
                return response_model.model_validate(data)

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(
                    f"Groq structured generation attempt {attempt}/{retries} failed validation: {e}. Retrying..."
                )
                # Append error feedback for next retry attempt
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response produced a validation error: {str(e)}. "
                            f"Please rectify the JSON according to the schema and return ONLY valid JSON."
                        ),
                    }
                )
                await asyncio.sleep(0.5 * attempt)

            except Exception as e:
                last_error = e
                logger.error(f"Groq API error on attempt {attempt}/{retries}: {e}")
                if attempt == retries:
                    raise
                await asyncio.sleep(1.0 * attempt)

        raise RuntimeError(
            f"Failed to generate valid structured response for {response_model.__name__} "
            f"after {retries} attempts: {last_error}"
        )


class MockLLMClient(BaseLLMClient):
    """Deterministic, grounded mock LLM client for testing and offline development.

    Extracts keywords and sentence summaries directly from the provided chunk text.
    Preserves exact source chunk IDs and page references passed in prompts.
    Supports deliberate failure simulation to test retry behavior and missing evidence handling.
    """

    def __init__(
        self,
        model_name: str = "mock-llama-3.3-70b",
        prompt_version: str = "v1.0",
        malformed_json_failures_remaining: int = 0,
    ):
        self._model_name = model_name
        self._prompt_version = prompt_version
        self.malformed_json_failures_remaining = malformed_json_failures_remaining
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        max_retries: Optional[int] = None,
    ) -> T:
        self.call_count += 1

        # Simulate malformed responses to test retry behavior
        if self.malformed_json_failures_remaining > 0:
            self.malformed_json_failures_remaining -= 1
            # Raise an error that simulates what GroqLLMClient catches during retry
            raise json.JSONDecodeError("Simulated malformed JSON response from LLM", doc="", pos=0)

        model_name = response_model.__name__

        # Extract any chunk IDs referenced in user_prompt
        import re
        chunk_ids = re.findall(r"\b[a-zA-Z0-9_-]+_c\d{4}\b", user_prompt)
        if not chunk_ids:
            chunk_ids = ["chunk_mock_c0001"]

        # Check if user_prompt signals sparse / insufficient text
        is_sparse = "sparse" in user_prompt.lower() or "insufficient" in user_prompt.lower() or len(user_prompt) < 100

        # Construct deterministic synthetic response based on target model
        if model_name == "SectionSummary":
            data = {
                "section_id": "sec_auto_01",
                "chapter_title": "Chapter 1",
                "section_title": "Section Overview",
                "page_start": 1,
                "page_end": 2,
                "concise_summary": f"Grounded summary based on evidence: {user_prompt[:120].strip()}...",
                "key_concepts": [
                    {
                        "name": "Core Principle",
                        "definition": "A foundational concept established directly in the text.",
                        "claim_type": "author_stated",
                        "source_chunk_ids": chunk_ids[:2],
                        "page_references": [1, 2],
                    }
                ],
                "definitions": [
                    {
                        "term": "Core Principle",
                        "explanation": "Formally defined principle governing the section.",
                        "verbatim_or_paraphrased": "paraphrased",
                        "source_chunk_id": chunk_ids[0],
                        "page_number": 1,
                    }
                ],
                "reasoning_chain": [
                    "Author introduces foundational premise.",
                    "Author substantiates premise with direct evidence.",
                ],
                "source_chunk_ids": chunk_ids,
                "uncertainty": "Evidence is sparse and requires cautious interpretation." if is_sparse else None,
                "prompt_version": self._prompt_version,
                "model_version": self._model_name,
            }
            return response_model.model_validate(data)

        elif model_name == "ChapterSummary":
            data = {
                "chapter_index": 0,
                "chapter_title": "Chapter 1",
                "page_start": 1,
                "page_end": 5,
                "chapter_crux": "The foundational crux: systems must be structured hierarchically to preserve evidence.",
                "central_argument": "Author argues that understanding requires layered abstraction anchored to concrete evidence.",
                "supporting_arguments": [
                    {
                        "claim": "Direct evidence must ground all high-level synthesis.",
                        "reasoning": "Without source chunk traceability, models hallucinate unsupported claims.",
                        "claim_type": "author_stated",
                        "source_chunk_ids": chunk_ids[:2],
                        "page_range": "pp. 1-3",
                    }
                ],
                "essential_concepts": [
                    {
                        "name": "Hierarchical Traceability",
                        "definition": "The uninterrupted provenance chain connecting words to thesis.",
                        "claim_type": "author_stated",
                        "source_chunk_ids": chunk_ids[:1],
                        "page_references": [1],
                    }
                ],
                "important_assumptions": [
                    {
                        "assumption": "Readers possess foundational literacy in the subject matter.",
                        "explicit_or_implicit": "implicit",
                        "source_chunk_ids": chunk_ids[:1],
                    }
                ],
                "examples": [
                    {
                        "description": "Case study demonstrating loss of nuance in ungrounded summaries.",
                        "is_counterexample": False,
                        "point_illustrated": "Illustrates why verbatim chunk tracking is necessary.",
                        "source_chunk_ids": chunk_ids[:1],
                    }
                ],
                "counterexamples": [],
                "conclusions": [
                    {
                        "statement": "Knowledge structures must retain leaf-level evidence pointers.",
                        "claim_type": "author_stated",
                        "source_chunk_ids": chunk_ids[:1],
                    }
                ],
                "prerequisites_and_connections": [
                    {
                        "related_chapter_title": "Chapter 2",
                        "nature_of_connection": "Prerequisite conceptual foundation for Chapter 2.",
                        "claim_type": "model_inferred",
                        "source_chunk_ids": chunk_ids[:1],
                    }
                ],
                "supporting_section_ids": ["sec_auto_01"],
                "source_chunk_ids": chunk_ids,
                "evidence_confidence": "insufficient" if is_sparse else "high",
                "uncertainty": "Chapter text is extremely brief or lacking empirical depth." if is_sparse else None,
                "stage_count": 1,
                "prompt_version": self._prompt_version,
                "model_version": self._model_name,
            }
            return response_model.model_validate(data)

        elif model_name == "BookSummary":
            data = {
                "book_id": "book_auto",
                "central_thesis": "Comprehensive book thesis: Deep understanding demands multi-tier synthesis anchored to empirical evidence.",
                "overall_crux": "Extract the crux without losing the evidence: four levels of knowledge preserve truth while surfacing wisdom.",
                "recurring_themes": [
                    {
                        "theme": "Evidence Preservation",
                        "manifestations": "Appears across all chapters as the guiding metric of truth.",
                        "chapters_present": ["Chapter 1"],
                        "source_chunk_ids": chunk_ids[:2],
                    }
                ],
                "essential_takeaways": [
                    {
                        "point": f"Takeaway #{i + 1}: Grounded reasoning ensures accuracy.",
                        "explanation": f"Detailed reasoning for takeaway #{i + 1} supported by source evidence.",
                        "claim_type": "author_stated" if i % 2 == 0 else "model_inferred",
                        "source_chunk_ids": chunk_ids[:1],
                        "page_numbers": [i + 1],
                    }
                    for i in range(5)
                ],
                "chapter_relationships": [
                    {
                        "source_chapter": "Chapter 1",
                        "target_chapter": "Chapter 2",
                        "relationship_type": "builds_upon",
                        "description": "Chapter 1 establishes the definitions that Chapter 2 applies.",
                        "claim_type": "model_inferred",
                        "source_chunk_ids": chunk_ids[:1],
                    }
                ],
                "uncertainty": "Some chapters contained sparse evidence." if is_sparse else None,
                "supporting_chapter_titles": ["Chapter 1"],
                "source_chunk_ids": chunk_ids,
                "prompt_version": self._prompt_version,
                "model_version": self._model_name,
            }
            return response_model.model_validate(data)

        elif model_name == "StructuredAnswer":
            # Check for unanswerable questions or explicit insufficient evidence indicators
            query_is_unanswerable = (
                "chocolate cake" in user_prompt.lower()
                or "recipe" in user_prompt.lower()
                or "unanswerable" in user_prompt.lower()
                or "not in book" in user_prompt.lower()
                or "baking" in user_prompt.lower()
                or "cookie" in user_prompt.lower()
                or ("quantum teleportation" in user_prompt.lower() and "quantum" not in user_prompt.lower())
            )

            if query_is_unanswerable or is_sparse:
                data = {
                    "crux": "The provided text from this book does not contain sufficient evidence to answer this question.",
                    "detailed_synthesis": (
                        "After reviewing the retrieved passages from the book, no evidence or discussion "
                        "regarding this subject was found. The book's scope does not cover this topic."
                    ),
                    "citations": [],
                    "insufficient_evidence": True,
                    "confidence": "insufficient",
                }
            else:
                # Extract page numbers from prompt if present
                page_matches = re.findall(r"Pages?\s+(\d+)(?:-(\d+))?", user_prompt, re.IGNORECASE)
                first_page = int(page_matches[0][0]) if page_matches else 1
                last_page = int(page_matches[0][1]) if page_matches and page_matches[0][1] else first_page

                # Extract grounding chunk texts from prompt if available
                evidence_texts = re.findall(
                    r"\[DOCUMENT EVIDENCE CHUNK [^\]]+\]\n(.*?)\n\[END EVIDENCE CHUNK",
                    user_prompt,
                    re.DOTALL,
                )
                if not evidence_texts:
                    evidence_texts = re.findall(
                        r"\[CHUNK [^\]]+\]:\s*(.*?)(?=\n\n\[CHUNK|\n\nQUESTION:|$)",
                        user_prompt,
                        re.DOTALL,
                    )

                sample_evidence = evidence_texts[0].strip() if evidence_texts else ""
                clean_snippet = sample_evidence.replace("\n", " ").strip()
                first_sentence = clean_snippet.split(".")[0] if clean_snippet else "the system operates through systematic verification"

                citations = [
                    {
                        "chunk_id": cid,
                        "page_start": first_page,
                        "page_end": last_page,
                        "chapter": "Chapter 1",
                        "section": "Section 1.1",
                        "quoted_snippet": clean_snippet[:120] if clean_snippet else "Relevant passage text directly substantiating the claim.",
                    }
                    for cid in chunk_ids[:3]
                ]

                is_quiz = "quiz" in user_prompt.lower() or "test me" in user_prompt.lower()
                is_comparison = "compare" in user_prompt.lower() or "versus" in user_prompt.lower() or "vs" in user_prompt.lower()

                if is_quiz:
                    crux = "Quiz Question: Based on the text, what is the primary operating mechanism?"
                    detail = (
                        "Question: What is the primary operating mechanism described in the text?\n\n"
                        "A) Deterministic execution\n"
                        "B) Adaptive feedback loop\n"
                        "C) Random walk\n\n"
                        f"Correct Answer: B (Adaptive feedback loop). As stated on page {first_page}, "
                        f"{first_sentence}. The system operates via recursive feedback loops rather than static execution."
                    )
                elif is_comparison:
                    crux = "Comparative Crux: The author contrasts both principles by their foundational assumptions and scaling properties."
                    detail = (
                        f"According to the source passages (pp. {first_page}-{last_page}), both concepts address structural organization, "
                        f"where {first_sentence.lower()}. In contrast to simple single-node designs, "
                        "both approaches differ in execution: the first operates via decentralized consensus whereas the second relies on hierarchical control. "
                        "Source claims explicitly highlight this distinction while model interpretation notes their complementary utility."
                    )
                else:
                    crux = "Direct Crux: The author's central finding directly addresses the inquiry by establishing foundational evidence."
                    detail = (
                        f"Based on the source evidence on page {first_page} (chunk {chunk_ids[0]}), the author explicitly states "
                        f"that {first_sentence.lower()}. Furthermore, multi-passage synthesis reveals "
                        f"that this directly aligns with the broader thesis established on page {last_page}."
                    )

                data = {
                    "crux": crux,
                    "detailed_synthesis": detail,
                    "citations": citations,
                    "insufficient_evidence": False,
                    "confidence": "high",
                }
            return response_model.model_validate(data)

        # Fallback empty model validation
        return response_model.model_validate({})


_llm_client_instance: Optional[BaseLLMClient] = None


def get_llm_client(force_new: bool = False, provider: Optional[str] = None) -> BaseLLMClient:
    """Factory returning configured LLM client singleton or new instance."""
    global _llm_client_instance
    if force_new or _llm_client_instance is None:
        chosen_provider = (provider or settings.LLM_PROVIDER).lower().strip()
        if chosen_provider == "ollama":
            _llm_client_instance = OllamaLLMClient(
                base_url=settings.OLLAMA_BASE_URL,
                model_name=settings.LLM_MODEL or settings.OLLAMA_MODEL,
                prompt_version=settings.PROMPT_VERSION,
                temperature=settings.OLLAMA_TEMPERATURE,
                timeout=settings.OLLAMA_TIMEOUT_SECONDS,
                num_ctx=settings.OLLAMA_NUM_CTX,
                num_predict=settings.OLLAMA_NUM_PREDICT,
                keep_alive=settings.OLLAMA_KEEP_ALIVE,
                max_retries=settings.LLM_MAX_RETRIES,
            )
        elif chosen_provider == "groq":
            if not settings.GROQ_API_KEY:
                raise RuntimeError(
                    "GROQ_API_KEY is not configured. Set GROQ_API_KEY in backend/.env "
                    "or configure LLM_PROVIDER=ollama for local inference."
                )
            _llm_client_instance = GroqLLMClient(
                api_key=settings.GROQ_API_KEY,
                model_name=settings.LLM_MODEL or settings.GROQ_MODEL,
                prompt_version=settings.PROMPT_VERSION,
                temperature=settings.LLM_TEMPERATURE,
                timeout=settings.LLM_TIMEOUT_SECONDS,
            )
        elif chosen_provider == "mock":
            _llm_client_instance = MockLLMClient(
                model_name=settings.LLM_MODEL or "mock-llama-3.3-70b",
                prompt_version=settings.PROMPT_VERSION,
            )
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: '{chosen_provider}'. "
                "Supported options are 'ollama', 'groq', or 'mock'."
            )
    return _llm_client_instance


def reset_llm_client() -> None:
    """Reset cached LLM client singleton."""
    global _llm_client_instance
    _llm_client_instance = None

