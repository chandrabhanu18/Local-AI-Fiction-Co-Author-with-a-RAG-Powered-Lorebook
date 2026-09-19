"""
Ollama LLM Service for local text generation and streaming.
"""
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional, List
import httpx

logger = logging.getLogger(__name__)


class OllamaService:
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama3.1:8b"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    async def is_healthy(self) -> bool:
        """Check if Ollama service is reachable and responsive."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/")
                return res.status_code == 200 and "Ollama is running" in res.text
        except Exception:
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """List locally available models in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return data.get("models", [])
        except Exception as e:
            logger.warning(f"Failed to fetch models from Ollama: {e}")
        return []

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Send a generation request to Ollama and return the full generated response."""
        target_model = model or self.default_model
        
        # Build options
        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if top_p is not None:
            options["top_p"] = float(top_p)
        if repeat_penalty is not None:
            options["repeat_penalty"] = float(repeat_penalty)
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)

        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False
        }
        if system_prompt:
            payload["system"] = system_prompt
        if options:
            payload["options"] = options

        logger.info(f"Dispatching generation request to Ollama ({target_model}) with options: {options}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "").strip()
                else:
                    err_msg = f"Ollama returned HTTP {response.status_code}: {response.text}"
                    logger.error(err_msg)
                    raise RuntimeError(err_msg)
        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}.")
            # Provide fallback simulation if Ollama server is not running during local testing
            return self._mock_generation(prompt, options)
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Stream generated tokens asynchronously from Ollama."""
        target_model = model or self.default_model
        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if top_p is not None:
            options["top_p"] = float(top_p)
        if repeat_penalty is not None:
            options["repeat_penalty"] = float(repeat_penalty)
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)

        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": True
        }
        if system_prompt:
            payload["system"] = system_prompt
        if options:
            payload["options"] = options

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        yield f"[Error: Ollama HTTP {response.status_code}]"
                        return

                    async for line in response.aiter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                token = chunk.get("response", "")
                                if token:
                                    yield token
                                if chunk.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
        except httpx.ConnectError:
            # Stream mock fallback tokens if offline
            mock_text = self._mock_generation(prompt, options)
            for word in mock_text.split(" "):
                yield word + " "

    def _mock_generation(self, prompt: str, options: Dict[str, Any]) -> str:
        """Deterministic mock generator for offline/unit test execution."""
        temp = options.get("temperature", 0.7)
        
        # Extract canon context if present in RAG prompt
        if "--- CANON CONTEXT ---" in prompt and "--- END CANON CONTEXT ---" in prompt:
            start = prompt.find("--- CANON CONTEXT ---") + len("--- CANON CONTEXT ---")
            end = prompt.find("--- END CANON CONTEXT ---")
            canon_text = prompt[start:end].strip().replace("\n", " ")
            return (
                f"Drawing upon the ancient canon ({canon_text}), the chronicles unfold. "
                f"Shadows dance across the stone in the moonlight. [Generated with temperature {temp:.2f}]"
            )

        return (
            f"The narrative continues as the atmosphere shifts. "
            f"Responding to '{prompt.strip()[:100]}', shadows dance in the moonlight. "
            f"[Generated with temperature {temp:.2f}]"
        )
