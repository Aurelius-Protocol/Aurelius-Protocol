"""On-chain E2E tests for Aurelius Protocol.

These tests interact with real blockchain operations on testnet (subnet 290).
They require:
- A validator wallet with testnet TAO
- Network access to Bittensor testnet
- Docker services for collector API (optional for some tests)

Safety: Tests will fail immediately if mainnet is detected.
"""
