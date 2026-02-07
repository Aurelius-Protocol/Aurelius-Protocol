"""Dataset logger for collecting alignment data."""

import atexit
import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Full, Queue

import bittensor as bt
import requests
from opentelemetry.trace import SpanKind

from aurelius.shared.telemetry.otel_setup import get_tracer


@dataclass
class DatasetEntry:
    """Entry in the alignment dataset.

    Attributes:
        timestamp: ISO format timestamp
        prompt: Original prompt from miner
        response: OpenAI's response to the prompt
        danger_score: Combined danger score (primary validator's initial score)
        category_scores: Breakdown by moderation category
        accepted: Whether prompt was accepted (met threshold)
        miner_hotkey: Miner's hotkey
        miner_uid: Miner's UID (if available)
        miner_coldkey: Miner's coldkey (if available)
        validator_hotkey: Validator's hotkey
        validator_uid: Validator's UID (if available)
        validator_coldkey: Validator's coldkey (if available)
        consensus_votes: Consensus vote tally (e.g., "4/5")
        consensus_verified: Whether entry passed consensus verification
        validator_votes: List of validator vote details
        mean_danger_score: Mean danger score across all runs (distribution)
        std_dev_danger_score: Standard deviation of danger scores
        min_danger_score: Minimum danger score observed
        max_danger_score: Maximum danger score observed
        total_runs: Total number of runs across all validators
        validator_count: Number of validators that participated
        excluded_validators: List of validator hotkeys that were excluded
        exclusion_reasons: Dict mapping hotkey to exclusion reason
        model_name: Name of the model used (e.g., "gpt-4o")
        model_config: Model configuration parameters (temperature, max_tokens, etc.)
        timing_metrics: Timing information (API call duration, moderation, total)
        network_context: Network context (subnet_uid, block_height, stakes)
        prompt_embedding: Pre-computed embedding vector (384 dimensions)
        experiment_id: Experiment ID for per-experiment novelty pools (T087)
    """

    timestamp: str
    prompt: str
    response: str
    danger_score: float
    category_scores: dict[str, float]
    accepted: bool
    miner_hotkey: str | None = None
    miner_uid: int | None = None
    miner_coldkey: str | None = None
    validator_hotkey: str | None = None
    validator_uid: int | None = None
    validator_coldkey: str | None = None
    consensus_votes: str | None = None
    consensus_verified: bool = False
    validator_votes: list | None = None
    mean_danger_score: float | None = None
    std_dev_danger_score: float | None = None
    min_danger_score: float | None = None
    max_danger_score: float | None = None
    total_runs: int | None = None
    validator_count: int | None = None
    excluded_validators: list | None = None
    exclusion_reasons: dict | None = None
    model_name: str | None = None
    model_config: dict | None = None
    timing_metrics: dict | None = None
    network_context: dict | None = None
    prompt_embedding: list[float] | None = None
    experiment_id: str | None = None  # T087: Per-experiment tracking


class DatasetLogger:
    """Logger for alignment dataset with local backup and central API submission."""

    def __init__(
        self,
        local_path: str | None = None,
        central_api_endpoint: str | None = None,
        central_api_key: str | None = None,
        enable_local_backup: bool = True,
        wallet: "bt.Wallet | None" = None,
    ):
        """
        Initialize dataset logger.

        Args:
            local_path: Path for local JSON backup files
            central_api_endpoint: URL of central API for data collection
            central_api_key: API key for authentication
            enable_local_backup: Whether to save local backups
            wallet: Bittensor wallet for signing submissions (optional)
        """
        self.local_path = local_path
        self.central_api_endpoint = central_api_endpoint
        self.central_api_key = central_api_key
        self.enable_local_backup = enable_local_backup
        self.wallet = wallet

        # Create local directory if needed
        if self.enable_local_backup and self.local_path:
            Path(self.local_path).mkdir(parents=True, exist_ok=True)
            bt.logging.info(f"Dataset logger: Local backup enabled at {self.local_path}")

        # Queue for async processing (bounded to prevent memory exhaustion during API outages)
        from aurelius.shared.config import Config
        self.queue: Queue = Queue(maxsize=Config.DATASET_LOGGER_QUEUE_MAXSIZE)
        self.worker_thread = None
        self.running = False

        # Thread lock for atomic file writes
        self._file_write_lock = threading.Lock()

        # HTTP session for connection pooling
        self._session = requests.Session()

        # Submission statistics
        self.submissions_successful = 0
        self.submissions_failed = 0

        # Telemetry tracer - import Config here to avoid circular import
        from aurelius.shared.config import Config
        self._tracer = get_tracer("aurelius.dataset") if Config.TELEMETRY_ENABLED else None

        # Start background worker if we have a central API
        if self.central_api_endpoint:
            bt.logging.info(f"Dataset logger: Central API enabled at {self.central_api_endpoint}")
            self._start_worker()
            # Register atexit handler for graceful shutdown
            atexit.register(self.stop)
        else:
            bt.logging.info("Dataset logger: Central API not configured")

    def _start_worker(self):
        """Start background worker thread for async API submissions."""
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker,
            daemon=False,  # Changed to non-daemon for proper cleanup
            name="DatasetLoggerWorker"
        )
        self.worker_thread.start()
        bt.logging.info("Dataset logger: Background worker thread started")

    def _worker(self):
        """Background worker that processes the queue."""
        while self.running:
            try:
                # Get entry from queue (blocks with timeout)
                entry = self.queue.get(timeout=1.0)
                if entry is None:  # Poison pill to stop worker
                    bt.logging.info("Dataset logger: Received stop signal")
                    break

                bt.logging.debug("Dataset logger: Processing queued entry")
                # Try to submit to central API
                self._submit_to_api(entry)
                self.queue.task_done()

            except Exception as e:
                # Queue.get timeout is normal, other errors we log
                if "Empty" not in str(type(e).__name__):
                    bt.logging.error(f"Dataset logger worker error: {e}")

    def _submit_to_api(self, entry: DatasetEntry, max_retries: int = 3) -> bool:
        """
        Submit entry to central API with retry logic.

        Args:
            entry: Dataset entry to submit
            max_retries: Maximum number of retry attempts

        Returns:
            True if successful, False otherwise
        """
        if not self.central_api_endpoint:
            return False

        start_time = time.time()

        # Wrap with tracing span if enabled
        if self._tracer:
            with self._tracer.start_as_current_span(
                "dataset.api_submit",
                kind=SpanKind.CLIENT,
                attributes={
                    "dataset.endpoint": self.central_api_endpoint,
                    "dataset.accepted": entry.accepted,
                    "dataset.danger_score": entry.danger_score,
                    "dataset.consensus_verified": entry.consensus_verified,
                },
            ) as span:
                success, status_code, retry_count = self._do_submit_to_api(entry, max_retries)
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("duration_ms", round(duration_ms, 2))
                span.set_attribute("dataset.success", success)
                span.set_attribute("dataset.retry_count", retry_count)
                if status_code:
                    span.set_attribute("http.status_code", status_code)
                return success
        else:
            success, _, _ = self._do_submit_to_api(entry, max_retries)
            return success

    def _do_submit_to_api(self, entry: DatasetEntry, max_retries: int) -> tuple[bool, int | None, int]:
        """Internal method to perform the API submission with retries."""
        headers = {
            "Content-Type": "application/json",
        }

        if self.central_api_key:
            headers["Authorization"] = f"Bearer {self.central_api_key}"

        # Serialize body once — used for both hashing and sending to ensure consistency
        data = asdict(entry)
        body_json = json.dumps(data, separators=(',', ':'), sort_keys=True)

        # Add wallet signature for authenticated submissions
        if self.wallet:
            try:
                timestamp = int(time.time())
                hotkey = self.wallet.hotkey.ss58_address
                # Compute body hash to bind signature to payload (prevents replay with different body)
                body_hash = hashlib.sha256(body_json.encode()).hexdigest()[:16]
                message = f"aurelius-submission:{timestamp}:{hotkey}:{body_hash}"
                signature = self.wallet.hotkey.sign(message.encode()).hex()
                headers.update({
                    "X-Validator-Hotkey": hotkey,
                    "X-Signature": signature,
                    "X-Timestamp": str(timestamp),
                    "X-Body-Hash": body_hash,
                })
            except Exception as e:
                bt.logging.warning(f"Failed to sign dataset submission: {e}")

        last_status_code = None

        for attempt in range(max_retries):
            try:
                # Send body_json directly (not json=data) to ensure the exact bytes
                # that were hashed are what the server receives
                response = self._session.post(
                    self.central_api_endpoint,
                    data=body_json,
                    headers=headers,
                    timeout=10,
                )
                last_status_code = response.status_code

                if response.status_code in [200, 201]:
                    self.submissions_successful += 1
                    bt.logging.success(f"Dataset entry submitted to central API (total: {self.submissions_successful})")
                    return True, last_status_code, attempt
                else:
                    bt.logging.warning(f"Central API returned status {response.status_code}: {response.text}")

            except requests.RequestException as e:
                bt.logging.warning(f"Failed to submit to central API (attempt {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff

        self.submissions_failed += 1
        bt.logging.error(f"Failed to submit to central API after all retries (total failures: {self.submissions_failed})")
        return False, last_status_code, max_retries - 1

    def _save_local(self, entry: DatasetEntry) -> bool:
        """
        Save entry to local JSON file using atomic write pattern.

        Creates one file per day in YYYY-MM-DD.jsonl format.
        Uses thread lock and temp file to prevent corruption from concurrent writes.

        Args:
            entry: Dataset entry to save

        Returns:
            True if successful, False otherwise
        """
        if not self.enable_local_backup or not self.local_path:
            return False

        try:
            # Create filename based on date
            date_str = datetime.now().strftime("%Y-%m-%d")
            filepath = Path(self.local_path) / f"{date_str}.jsonl"

            # Serialize entry to JSON string first (outside lock)
            json_line = json.dumps(asdict(entry)) + "\n"

            # Use lock to prevent concurrent writes corrupting the file
            with self._file_write_lock:
                # Write to temp file first, then append atomically
                # This prevents partial writes from corrupting the file
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    dir=self.local_path,
                    suffix='.tmp',
                    delete=False
                ) as tmp_file:
                    tmp_file.write(json_line)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())  # Ensure data is on disk
                    tmp_path = tmp_file.name

                # Append temp file contents to main file
                with open(filepath, "a") as main_file:
                    with open(tmp_path, "r") as tmp_read:
                        main_file.write(tmp_read.read())
                    main_file.flush()
                    os.fsync(main_file.fileno())

                # Clean up temp file
                os.unlink(tmp_path)

            return True

        except Exception as e:
            bt.logging.error(f"Failed to save local backup: {e}")
            # Clean up temp file on error
            try:
                if 'tmp_path' in locals():
                    os.unlink(tmp_path)
            except:
                pass
            return False

    def log_entry(
        self,
        prompt: str,
        response: str,
        danger_score: float,
        category_scores: dict[str, float],
        accepted: bool,
        miner_hotkey: str | None = None,
        miner_uid: int | None = None,
        miner_coldkey: str | None = None,
        validator_hotkey: str | None = None,
        validator_uid: int | None = None,
        validator_coldkey: str | None = None,
        consensus_votes: str | None = None,
        consensus_verified: bool = False,
        validator_votes: list | None = None,
        mean_danger_score: float | None = None,
        std_dev_danger_score: float | None = None,
        min_danger_score: float | None = None,
        max_danger_score: float | None = None,
        total_runs: int | None = None,
        validator_count: int | None = None,
        excluded_validators: list | None = None,
        exclusion_reasons: dict | None = None,
        model_name: str | None = None,
        model_config: dict | None = None,
        timing_metrics: dict | None = None,
        network_context: dict | None = None,
        prompt_embedding: list[float] | None = None,
        experiment_id: str | None = None,
    ) -> None:
        """
        Log a dataset entry.

        This method is non-blocking - it saves locally immediately
        and queues the central API submission for background processing.

        Args:
            prompt: Original prompt from miner
            response: OpenAI's response
            danger_score: Combined danger score (initial)
            category_scores: Category breakdown
            accepted: Whether prompt met threshold
            miner_hotkey: Miner's hotkey
            miner_uid: Miner's UID
            miner_coldkey: Miner's coldkey
            validator_hotkey: Validator's hotkey
            validator_uid: Validator's UID
            validator_coldkey: Validator's coldkey
            consensus_votes: Consensus vote tally
            consensus_verified: Whether consensus was reached
            validator_votes: Detailed validator vote information
            mean_danger_score: Mean danger score from distribution
            std_dev_danger_score: Std dev of danger scores
            min_danger_score: Minimum danger score
            max_danger_score: Maximum danger score
            total_runs: Total number of runs
            validator_count: Number of validators that participated
            excluded_validators: List of excluded validators
            exclusion_reasons: Reasons for exclusions
            model_name: Name of the model used
            model_config: Model configuration parameters
            timing_metrics: Timing information
            network_context: Network context data
            prompt_embedding: Pre-computed embedding vector (384 dimensions)
        """
        entry = DatasetEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt=prompt,
            response=response,
            danger_score=danger_score,
            category_scores=category_scores,
            accepted=accepted,
            miner_hotkey=miner_hotkey,
            miner_uid=miner_uid,
            miner_coldkey=miner_coldkey,
            validator_hotkey=validator_hotkey,
            validator_uid=validator_uid,
            validator_coldkey=validator_coldkey,
            consensus_votes=consensus_votes,
            consensus_verified=consensus_verified,
            validator_votes=validator_votes,
            mean_danger_score=mean_danger_score,
            std_dev_danger_score=std_dev_danger_score,
            min_danger_score=min_danger_score,
            max_danger_score=max_danger_score,
            total_runs=total_runs,
            validator_count=validator_count,
            excluded_validators=excluded_validators,
            exclusion_reasons=exclusion_reasons,
            model_name=model_name,
            model_config=model_config,
            timing_metrics=timing_metrics,
            network_context=network_context,
            prompt_embedding=prompt_embedding,
            experiment_id=experiment_id,  # T087: Per-experiment tracking
        )

        # Save locally (blocking, fast)
        if self.enable_local_backup:
            self._save_local(entry)

        # Queue for central API submission (non-blocking)
        if self.central_api_endpoint:
            try:
                self.queue.put_nowait(entry)
                bt.logging.debug(f"Dataset logger: Entry queued (queue size: {self.queue.qsize()})")
            except Full:
                bt.logging.warning(
                    f"Dataset logger: Queue full ({self.queue.maxsize} entries), dropping entry. "
                    "Central API may be slow or down."
                )

    def stop(self):
        """Stop the background worker thread with queue persistence."""
        if not self.worker_thread or not self.running:
            return  # Already stopped

        from aurelius.shared.config import Config

        queue_size = self.queue.qsize()
        if queue_size > 0:
            bt.logging.info(f"Stopping dataset logger worker... ({queue_size} entries pending)")
        else:
            bt.logging.info("Stopping dataset logger worker...")

        self.running = False

        # Wait for queue to empty (with timeout)
        timeout = Config.DATASET_LOGGER_SHUTDOWN_TIMEOUT
        bt.logging.info(f"Waiting up to {timeout}s for queue to drain...")

        # Use a shorter timeout per item to allow checking progress
        items_processed = 0
        remaining_items = queue_size

        while remaining_items > 0 and items_processed < queue_size:
            try:
                self.queue.join()  # This will return when queue is empty
                break
            except:
                pass
            remaining_items = self.queue.qsize()
            items_processed = queue_size - remaining_items

        # Send poison pill
        self.queue.put(None)

        # Wait for thread to finish
        self.worker_thread.join(timeout=min(timeout, 10))

        if self.worker_thread.is_alive():
            bt.logging.warning(
                f"Dataset logger: Worker thread did not stop within {timeout}s timeout"
            )
            # Persist remaining queue items to disk
            remaining = self.queue.qsize()
            if remaining > 0:
                self._persist_queue()
        else:
            bt.logging.info("Dataset logger worker stopped cleanly")

    def _persist_queue(self):
        """Persist remaining queue items to disk to prevent data loss."""
        if not self.enable_local_backup or not self.local_path:
            bt.logging.warning("Cannot persist queue - local backup not enabled")
            return

        try:
            remaining_items = []
            while not self.queue.empty():
                try:
                    entry = self.queue.get_nowait()
                    if entry is not None:  # Skip poison pill
                        remaining_items.append(entry)
                except:
                    break

            if remaining_items:
                persist_file = Path(self.local_path) / f"unsent_queue_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
                with open(persist_file, "w") as f:
                    json.dump([asdict(entry) for entry in remaining_items], f, indent=2)
                bt.logging.warning(
                    f"Persisted {len(remaining_items)} unsent entries to {persist_file}. "
                    f"These were not submitted to the central API."
                )
        except Exception as e:
            bt.logging.error(f"Failed to persist queue: {e}")

    def get_stats(self) -> dict:
        """
        Get logger statistics.

        Returns:
            Dictionary with current stats
        """
        return {
            "local_backup_enabled": self.enable_local_backup,
            "local_path": self.local_path,
            "central_api_enabled": self.central_api_endpoint is not None,
            "queue_size": self.queue.qsize() if self.central_api_endpoint else 0,
            "submissions_successful": self.submissions_successful,
            "submissions_failed": self.submissions_failed,
        }
