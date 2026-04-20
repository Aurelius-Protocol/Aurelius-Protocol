"""Local storage for scenario configs served by the miner.

Reads JSON files from a configurable directory and serves them round-robin.
"""

import json
import logging
from pathlib import Path

from aurelius.common.schema import validate_scenario_config

logger = logging.getLogger(__name__)


class ConfigStore:
    def __init__(self, config_dir: str | Path):
        self.config_dir = Path(config_dir)
        self.configs: list[dict] = []
        self._index = 0
        self._load_configs()

    def _load_configs(self) -> None:
        if not self.config_dir.exists():
            logger.warning("Config directory does not exist: %s", self.config_dir)
            return

        for path in sorted(self.config_dir.glob("*.json")):
            try:
                with open(path) as f:
                    config = json.load(f)
                result = validate_scenario_config(config)
                if result.valid:
                    self.configs.append(config)
                    logger.info("Loaded config: %s", path.name)
                else:
                    logger.warning("Invalid config %s: %s", path.name, result.errors)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load %s: %s", path.name, e)

        logger.info("Loaded %d scenario configs from %s", len(self.configs), self.config_dir)

    def next(self) -> dict | None:
        """Return the next config in round-robin order, or None if empty."""
        if not self.configs:
            return None
        config = self.configs[self._index % len(self.configs)]
        self._index += 1
        return config

    def reload(self) -> None:
        """Reload configs from disk."""
        self.configs.clear()
        self._index = 0
        self._load_configs()

    @property
    def count(self) -> int:
        return len(self.configs)
