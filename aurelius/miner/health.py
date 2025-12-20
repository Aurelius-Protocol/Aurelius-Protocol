"""Pre-flight health checks for validator connectivity."""

import socket
import time
from dataclasses import dataclass

from aurelius.miner.errors import (
    ErrorCategory,
    MinerError,
    classify_error,
    create_error,
)


@dataclass
class HealthCheckResult:
    """Result of a validator health check."""

    is_healthy: bool
    latency_ms: float | None = None
    error: MinerError | None = None
    details: dict | None = None


class ValidatorHealthChecker:
    """Pre-flight validator connectivity checker."""

    def __init__(self, timeout: float = 5.0):
        """
        Initialize health checker.

        Args:
            timeout: Timeout in seconds for health checks
        """
        self.timeout = timeout

    def check_dns_resolution(self, host: str) -> HealthCheckResult:
        """
        Verify DNS resolution works for the host.

        Args:
            host: Hostname to resolve

        Returns:
            HealthCheckResult with resolution status
        """
        # Skip DNS check for IP addresses
        if self._is_ip_address(host):
            return HealthCheckResult(
                is_healthy=True,
                details={"host": host, "type": "ip_address"},
            )

        context = {"host": host}
        start_time = time.time()

        try:
            socket.getaddrinfo(host, None, socket.AF_INET)
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                is_healthy=True,
                latency_ms=latency_ms,
                details={"host": host, "type": "dns_resolved"},
            )
        except socket.gaierror as e:
            return HealthCheckResult(
                is_healthy=False,
                error=create_error(ErrorCategory.DNS_RESOLUTION, context, e),
                details={"host": host},
            )
        except Exception as e:
            return HealthCheckResult(
                is_healthy=False,
                error=classify_error(e, context),
                details={"host": host},
            )

    def check_tcp_connectivity(self, host: str, port: int) -> HealthCheckResult:
        """
        Test TCP connection to host:port.

        Args:
            host: Host to connect to
            port: Port to connect to

        Returns:
            HealthCheckResult with connectivity status
        """
        context = {"host": host, "port": port}
        start_time = time.time()

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)

            result = sock.connect_ex((host, port))
            latency_ms = (time.time() - start_time) * 1000
            sock.close()

            if result == 0:
                return HealthCheckResult(
                    is_healthy=True,
                    latency_ms=latency_ms,
                    details={"host": host, "port": port},
                )
            else:
                # Connection failed with error code
                error_msg = self._get_socket_error_message(result)
                context["error"] = error_msg

                if result in (111, 10061):  # Connection refused (Linux/Windows)
                    error = create_error(
                        ErrorCategory.CONNECTION_REFUSED,
                        context,
                        OSError(result, error_msg),
                    )
                elif result in (110, 10060):  # Connection timed out
                    context["timeout"] = self.timeout
                    error = create_error(
                        ErrorCategory.NETWORK_TIMEOUT,
                        context,
                        OSError(result, error_msg),
                    )
                else:
                    error = create_error(
                        ErrorCategory.CONNECTION_REFUSED,
                        context,
                        OSError(result, error_msg),
                    )

                return HealthCheckResult(
                    is_healthy=False,
                    error=error,
                    details={"host": host, "port": port, "errno": result},
                )

        except TimeoutError:
            context["timeout"] = self.timeout
            return HealthCheckResult(
                is_healthy=False,
                error=create_error(
                    ErrorCategory.NETWORK_TIMEOUT,
                    context,
                    TimeoutError(f"Connection timed out after {self.timeout}s"),
                ),
                details={"host": host, "port": port},
            )
        except ConnectionRefusedError as e:
            return HealthCheckResult(
                is_healthy=False,
                error=create_error(ErrorCategory.CONNECTION_REFUSED, context, e),
                details={"host": host, "port": port},
            )
        except Exception as e:
            return HealthCheckResult(
                is_healthy=False,
                error=classify_error(e, context),
                details={"host": host, "port": port},
            )

    def check_validator_reachable(
        self,
        host: str,
        port: int,
    ) -> HealthCheckResult:
        """
        Perform full validator reachability check.

        This performs:
        1. DNS resolution (if hostname)
        2. TCP connectivity test

        Args:
            host: Validator host (IP or hostname)
            port: Validator port

        Returns:
            HealthCheckResult with overall status
        """
        # Step 1: DNS resolution (if hostname)
        if not self._is_ip_address(host):
            dns_result = self.check_dns_resolution(host)
            if not dns_result.is_healthy:
                return dns_result

        # Step 2: TCP connectivity
        tcp_result = self.check_tcp_connectivity(host, port)
        return tcp_result

    def _is_ip_address(self, host: str) -> bool:
        """Check if host is an IP address (not hostname)."""
        try:
            socket.inet_aton(host)
            return True
        except OSError:
            pass

        # Check IPv6
        try:
            socket.inet_pton(socket.AF_INET6, host)
            return True
        except OSError:
            pass

        return False

    def _get_socket_error_message(self, errno: int) -> str:
        """Get human-readable error message for socket error code."""
        error_messages = {
            # Linux/macOS error codes
            111: "Connection refused",
            110: "Connection timed out",
            113: "No route to host",
            101: "Network is unreachable",
            # Windows error codes
            10061: "Connection refused",
            10060: "Connection timed out",
            10065: "No route to host",
            10051: "Network is unreachable",
        }
        return error_messages.get(errno, f"Socket error {errno}")
