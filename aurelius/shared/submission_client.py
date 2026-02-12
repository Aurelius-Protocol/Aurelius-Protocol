"""Submission client for async token-based submission tracking via collector API."""

import hashlib
import json
import time

import bittensor as bt
import requests

from aurelius.shared.config import Config


class SubmissionClient:
    """HTTP client for the collector API's submissions endpoints.

    Follows the same SR25519 signing pattern as dataset_logger.py.
    """

    def __init__(
        self,
        base_url: str | None = None,
        wallet: "bt.Wallet | None" = None,
        api_key: str | None = None,
    ):
        self._base_url = (base_url or self._derive_base_url()).rstrip("/")
        self._wallet = wallet
        self._api_key = api_key
        self._session = requests.Session()

    @staticmethod
    def _derive_base_url() -> str:
        """Derive submissions endpoint from CENTRAL_API_ENDPOINT."""
        endpoint = Config.CENTRAL_API_ENDPOINT or ""
        # CENTRAL_API_ENDPOINT ends with /api/collections — replace path
        if "/api/" in endpoint:
            base = endpoint[: endpoint.index("/api/")]
        else:
            base = endpoint.rstrip("/")
        return f"{base}/api/submissions"

    def _build_headers(self, body_json: str | None = None) -> dict[str, str]:
        """Build request headers with optional SR25519 signing."""
        headers = {"Content-Type": "application/json"}

        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        if self._wallet:
            try:
                timestamp = int(time.time())
                hotkey = self._wallet.hotkey.ss58_address
                body_hash = ""
                if body_json:
                    body_hash = hashlib.sha256(body_json.encode()).hexdigest()
                    message = f"aurelius-submission:{timestamp}:{hotkey}:{body_hash}"
                else:
                    message = f"aurelius-submission:{timestamp}:{hotkey}"
                signature = self._wallet.hotkey.sign(message.encode()).hex()
                headers.update(
                    {
                        "X-Validator-Hotkey": hotkey,
                        "X-Signature": signature,
                        "X-Timestamp": str(timestamp),
                    }
                )
                if body_hash:
                    headers["X-Body-Hash"] = body_hash
            except Exception as e:
                bt.logging.warning(f"Failed to sign submission request: {e}")

        return headers

    def register_submission(
        self,
        token: str,
        miner_hotkey: str,
        experiment_id: str,
        prompt_hash: str | None = None,
    ) -> bool:
        """Register a new submission token on the collector API.

        Returns True if successful, False otherwise.
        """
        data = {
            "submission_token": token,
            "miner_hotkey": miner_hotkey,
            "experiment_id": experiment_id,
        }
        if prompt_hash:
            data["prompt_hash"] = prompt_hash

        body_json = json.dumps(data, separators=(",", ":"), sort_keys=True)
        headers = self._build_headers(body_json)

        try:
            response = self._session.post(
                self._base_url,
                data=body_json,
                headers=headers,
                timeout=10,
            )
            if response.status_code in (200, 201):
                return True
            bt.logging.warning(
                f"Failed to register submission token: {response.status_code} {response.text}"
            )
            return False
        except requests.RequestException as e:
            bt.logging.warning(f"Failed to register submission token: {e}")
            return False

    def update_submission(
        self,
        token: str,
        status: str,
        result: dict | None = None,
        execution_id: int | None = None,
        error: str | None = None,
    ) -> bool:
        """Update a submission's status on the collector API.

        Returns True if successful, False otherwise.
        """
        data: dict = {"status": status}
        if result is not None:
            data["result"] = result
        if execution_id is not None:
            data["execution_id"] = execution_id
        if error is not None:
            data["error_message"] = error

        body_json = json.dumps(data, separators=(",", ":"), sort_keys=True)
        headers = self._build_headers(body_json)

        try:
            response = self._session.put(
                f"{self._base_url}/{token}",
                data=body_json,
                headers=headers,
                timeout=10,
            )
            if response.status_code == 200:
                return True
            bt.logging.warning(
                f"Failed to update submission {token}: {response.status_code} {response.text}"
            )
            return False
        except requests.RequestException as e:
            bt.logging.warning(f"Failed to update submission {token}: {e}")
            return False

    def get_submission(self, token: str) -> dict | None:
        """Get submission status from collector API (fallback for validator restart).

        Returns the submission dict or None if not found.
        """
        try:
            response = self._session.get(
                f"{self._base_url}/{token}",
                timeout=10,
            )
            if response.status_code == 200:
                return response.json()
            return None
        except requests.RequestException as e:
            bt.logging.warning(f"Failed to get submission {token}: {e}")
            return None

    def close(self):
        """Close the HTTP session."""
        self._session.close()
