"""Base class for pull experiment handlers (T091).

Pull experiments are validator-initiated queries where validators request data
from registered miners on a schedule. Miners must implement handlers to respond
to these requests.

Example implementation:

    from aurelius.miner.pull_handler import PullHandler
    from aurelius.shared.protocol import PullRequestSynapse

    class MyDataHandler(PullHandler):
        def handle(self, synapse: PullRequestSynapse) -> PullRequestSynapse:
            # Process the query and prepare response
            data = self._fetch_data(synapse.query_params)

            # Populate response fields
            synapse.response_data = {"results": data}
            synapse.response_timestamp = datetime.utcnow().isoformat()

            return synapse

        def _fetch_data(self, params: dict) -> list:
            # Your data fetching logic here
            ...

Registration:
    Before your miner can receive pull requests, you must register for the
    experiment using the CLI:

        python miner.py --register-experiment <experiment_id>

    Verify registration:

        python miner.py --list-registrations

Timeout handling:
    - Validators have a configured timeout (default 30s)
    - If your handler doesn't respond in time, the request times out
    - Timeouts are recorded as non-responses and may affect scoring
    - Keep processing efficient to avoid timeouts
"""

from abc import ABC, abstractmethod
from datetime import datetime

import bittensor as bt

from aurelius.shared.protocol import PullRequestSynapse


class PullHandler(ABC):
    """Abstract base class for handling pull experiment requests.

    Miners must subclass this and implement the handle() method to
    participate in pull-based experiments.

    Attributes:
        experiment_id: The experiment this handler is registered for
        timeout_seconds: Maximum time allowed for handling a request
    """

    def __init__(self, experiment_id: str, timeout_seconds: int = 30):
        """Initialize pull handler.

        Args:
            experiment_id: Experiment ID this handler is for
            timeout_seconds: Maximum time to handle a request
        """
        self.experiment_id = experiment_id
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    def handle(self, synapse: PullRequestSynapse) -> PullRequestSynapse:
        """Handle an incoming pull request from a validator.

        This method MUST be implemented by subclasses. It receives the
        pull request synapse and should populate the response fields:
        - synapse.response_data: Dict with your response data
        - synapse.response_timestamp: ISO timestamp of when you responded

        Args:
            synapse: The incoming pull request from validator

        Returns:
            The synapse with response_data and response_timestamp populated

        Raises:
            Exception: If an error occurs, set synapse.error_message instead
                of raising. The synapse should still be returned.
        """
        pass

    def validate_request(self, synapse: PullRequestSynapse) -> str | None:
        """Validate an incoming pull request.

        Override this method to add custom validation.

        Args:
            synapse: The incoming pull request

        Returns:
            Error message if validation fails, None if valid
        """
        if synapse.experiment_id != self.experiment_id:
            return f"Handler for {self.experiment_id}, got request for {synapse.experiment_id}"

        if not synapse.request_id:
            return "Missing request_id"

        return None

    def process(self, synapse: PullRequestSynapse) -> PullRequestSynapse:
        """Process a pull request with validation and error handling.

        This is the main entry point called by the miner's axon.
        It validates the request, calls handle(), and handles errors.

        Args:
            synapse: The incoming pull request

        Returns:
            The processed synapse with response or error
        """
        # Validate request
        validation_error = self.validate_request(synapse)
        if validation_error:
            synapse.error_message = validation_error
            synapse.response_timestamp = datetime.utcnow().isoformat()
            bt.logging.warning(f"Pull request validation failed: {validation_error}")
            return synapse

        # Handle request
        try:
            result = self.handle(synapse)

            # Ensure timestamp is set
            if not result.response_timestamp:
                result.response_timestamp = datetime.utcnow().isoformat()

            bt.logging.debug(
                f"Pull request handled: experiment={self.experiment_id}, "
                f"request_id={synapse.request_id}"
            )
            return result

        except Exception as e:
            bt.logging.error(f"Pull handler error: {e}")
            synapse.error_message = str(e)
            synapse.response_timestamp = datetime.utcnow().isoformat()
            return synapse


class ExamplePullHandler(PullHandler):
    """Example implementation of a pull handler.

    This demonstrates how to implement a simple pull handler that
    echoes back the query parameters.

    Usage:
        handler = ExamplePullHandler("example-experiment")
        synapse = handler.process(incoming_synapse)
    """

    def handle(self, synapse: PullRequestSynapse) -> PullRequestSynapse:
        """Echo back query parameters as response.

        Args:
            synapse: The incoming pull request

        Returns:
            Synapse with echoed parameters
        """
        synapse.response_data = {
            "status": "ok",
            "echo": synapse.query_params or {},
            "query_type": synapse.query_type,
        }
        synapse.response_timestamp = datetime.utcnow().isoformat()
        return synapse
