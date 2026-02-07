"""Embedding client for generating text embeddings via OpenAI API."""

import time

import bittensor as bt
import requests
from opentelemetry.trace import SpanKind

from aurelius.shared.config import Config
from aurelius.shared.telemetry.otel_setup import get_tracer

# Embedding model details - OpenAI text-embedding-3-small supports dimension reduction
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 384  # Request 384 dimensions for storage efficiency
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


class EmbeddingClient:
    """Client for generating text embeddings using OpenAI's embedding API."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 30,
    ):
        """
        Initialize embedding client.

        Args:
            api_key: OpenAI API key (defaults to Config.OPENAI_API_KEY)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.timeout = timeout
        self.model = EMBEDDING_MODEL
        self.dimensions = EMBEDDING_DIMENSIONS
        self.embeddings_url = OPENAI_EMBEDDINGS_URL
        self._session = requests.Session()
        self._tracer = get_tracer("aurelius.embedding") if Config.TELEMETRY_ENABLED else None

        if self.api_key:
            bt.logging.info(f"Embedding client initialized: {self.embeddings_url} (model: {self.model})")
        else:
            bt.logging.warning("Embedding client: Missing OpenAI API key")

    def is_available(self) -> bool:
        """Check if embedding generation is available."""
        return bool(self.api_key)

    def _get_embeddings_url(self) -> str:
        """Get the full embeddings endpoint URL."""
        return self.embeddings_url

    def get_embedding(self, text: str) -> list[float] | None:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            List of floats (384 dimensions) or None on error
        """
        if not self.is_available():
            bt.logging.debug("Embedding generation skipped: client not configured")
            return None

        start_time = time.time()

        # Wrap with tracing span if enabled
        if self._tracer:
            with self._tracer.start_as_current_span(
                "embedding.generate",
                kind=SpanKind.CLIENT,
                attributes={
                    "embedding.model": self.model,
                    "embedding.dimensions": self.dimensions,
                    "embedding.text_length": len(text),
                    "http.url": self.embeddings_url,
                },
            ) as span:
                result = self._do_get_embedding(text)
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("duration_ms", round(duration_ms, 2))
                span.set_attribute("embedding.success", result is not None)
                return result
        else:
            return self._do_get_embedding(text)

    def _do_get_embedding(self, text: str) -> list[float] | None:
        """Internal method to perform the embedding API call."""
        try:
            response = self._session.post(
                self.embeddings_url,
                json={
                    "model": self.model,
                    "input": [text],
                    "dimensions": self.dimensions,  # Request specific dimensions
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                embedding = data["data"][0]["embedding"]
                if len(embedding) != self.dimensions:
                    raise ValueError(
                        f"Embedding dimension mismatch: got {len(embedding)}, expected {self.dimensions}"
                    )
                return embedding
            else:
                bt.logging.warning(f"Embedding API error: HTTP {response.status_code} - {response.text[:200]}")
                return None

        except requests.Timeout:
            bt.logging.warning(f"Embedding request timed out after {self.timeout}s")
            return None
        except requests.RequestException as e:
            bt.logging.warning(f"Embedding request failed: {e}")
            return None
        except (KeyError, IndexError) as e:
            bt.logging.warning(f"Embedding response parsing failed: {e}")
            return None
        except Exception as e:
            bt.logging.error(f"Embedding unexpected error: {e}")
            return None

    def get_embeddings(self, texts: list[str]) -> list[list[float]] | None:
        """
        Generate embeddings for multiple texts in a single request.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings or None on error
        """
        if not self.is_available():
            return None

        if not texts:
            return []

        try:
            response = self._session.post(
                self.embeddings_url,
                json={
                    "model": self.model,
                    "input": texts,
                    "dimensions": self.dimensions,
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                # Sort by index to ensure correct order
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
            else:
                bt.logging.warning(f"Batch embedding API error: HTTP {response.status_code}")
                return None

        except Exception as e:
            bt.logging.warning(f"Batch embedding failed: {e}")
            return None


# Singleton instance
_embedding_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    """Get singleton embedding client instance."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
