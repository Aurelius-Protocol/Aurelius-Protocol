"""Configuration management for the Aurelius miner."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Miner configuration loaded from environment variables."""

    # Miner Configuration
    MINER_WALLET_NAME: str = os.getenv("MINER_WALLET_NAME", "miner")
    MINER_HOTKEY: str = os.getenv("MINER_HOTKEY", "default")

    # Bittensor Network Configuration
    BT_NETWORK: str = os.getenv("BT_NETWORK", "finney")
    BT_NETUID: int = int(os.getenv("BT_NETUID", "1"))

    # Validator Configuration (for queries)
    BT_PORT_VALIDATOR: int = int(os.getenv("BT_PORT_VALIDATOR", "8091"))

    # Chain endpoint (optional - for custom subtensor endpoints)
    SUBTENSOR_ENDPOINT: str | None = os.getenv("SUBTENSOR_ENDPOINT")

    # Local Mode - Skip blockchain, connect directly to validator by IP
    # Set to true for testing without wallet registration/stake
    LOCAL_MODE: bool = os.getenv("LOCAL_MODE", "false").lower() == "true"
    VALIDATOR_HOST: str = os.getenv("VALIDATOR_HOST", "127.0.0.1")

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()  # DEBUG, INFO, WARNING, ERROR
    LOG_SENSITIVE_DATA: bool = os.getenv("LOG_SENSITIVE_DATA", "false").lower() == "true"
    MAX_LOG_LENGTH: int = int(os.getenv("MAX_LOG_LENGTH", "100"))

    @classmethod
    def get_subtensor_config(cls) -> dict:
        """Get subtensor configuration for network connection."""
        # If a custom endpoint is specified, use it as the network parameter
        # Otherwise use the BT_NETWORK value (local/test/finney)
        if cls.SUBTENSOR_ENDPOINT:
            config = {"network": cls.SUBTENSOR_ENDPOINT}
        else:
            config = {"network": cls.BT_NETWORK}
        return config

    @classmethod
    def setup_logging(cls) -> None:
        """Configure logging based on LOG_LEVEL setting."""
        import logging
        import bittensor as bt

        # Map string levels to logging constants
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }

        level = level_map.get(cls.LOG_LEVEL, logging.INFO)

        # Configure bittensor logging
        bt.logging.set_debug(level == logging.DEBUG)
        bt.logging.set_trace(level == logging.DEBUG)

        # Also configure root Python logger
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    @classmethod
    def truncate_sensitive_data(cls, text: str) -> str:
        """
        Truncate sensitive data for logging if LOG_SENSITIVE_DATA is False.

        Args:
            text: The text to potentially truncate

        Returns:
            str: Original text if logging enabled, truncated text otherwise
        """
        if cls.LOG_SENSITIVE_DATA:
            return text

        if len(text) <= cls.MAX_LOG_LENGTH:
            return text

        return text[:cls.MAX_LOG_LENGTH] + "... [TRUNCATED]"
