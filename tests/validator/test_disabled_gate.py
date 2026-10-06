"""VALIDATOR_ENABLED master switch: off by default, local-only.

A disabled validator must never construct Validator (wallet, chain,
Central API, set_weights), and must exit promptly on SIGTERM.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from unittest.mock import patch

from aurelius.validator import validator as validator_mod


class TestMainGate:
    def test_disabled_never_constructs_validator(self):
        with (
            patch.object(validator_mod.Config, "VALIDATOR_ENABLED", False),
            patch.object(validator_mod, "_configure_logging"),
            patch.object(validator_mod, "_idle_while_disabled") as idle,
            patch.object(validator_mod, "Validator") as validator_cls,
            patch.object(validator_mod.asyncio, "run") as run,
            patch("sys.argv", ["aurelius-validator"]),
        ):
            validator_mod.main()
        idle.assert_called_once()
        validator_cls.assert_not_called()
        run.assert_not_called()

    def test_enabled_runs_validator(self):
        with (
            patch.object(validator_mod.Config, "VALIDATOR_ENABLED", True),
            patch.object(validator_mod, "_configure_logging"),
            patch.object(validator_mod, "_idle_while_disabled") as idle,
            patch.object(validator_mod, "Validator") as validator_cls,
            patch.object(validator_mod.asyncio, "run") as run,
            patch("sys.argv", ["aurelius-validator"]),
        ):
            validator_mod.main()
        idle.assert_not_called()
        validator_cls.assert_called_once()
        run.assert_called_once()


class TestIdleWhileDisabled:
    def test_logs_disabled_warning(self, caplog):
        # Stop the loop after its first sleep.
        with (
            patch.object(validator_mod.time, "sleep", side_effect=SystemExit(0)),
            patch.object(validator_mod.signal, "signal"),
            caplog.at_level(logging.WARNING, logger=validator_mod.logger.name),
        ):
            try:
                validator_mod._idle_while_disabled(interval=0.01)
            except SystemExit:
                pass
        assert any("Validator is DISABLED" in r.message for r in caplog.records)

    def test_sigterm_exits_promptly(self):
        # Real process + real signal: the handler path is what lets
        # `docker stop` / Watchtower updates finish without SIGKILL.
        env = {k: v for k, v in os.environ.items() if k != "VALIDATOR_ENABLED"}
        env["VALIDATOR_ENABLED"] = "0"
        proc = subprocess.Popen(
            [sys.executable, "-c", "from aurelius.validator.validator import main; main()"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            deadline = time.monotonic() + 60
            line = ""
            while "Validator is DISABLED" not in line:
                assert time.monotonic() < deadline, "validator never reached the idle loop"
                line = proc.stdout.readline()
                assert line or proc.poll() is None, "validator exited before idling"
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(timeout=5) == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
