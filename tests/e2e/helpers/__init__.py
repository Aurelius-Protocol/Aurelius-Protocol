"""Helper utilities for E2E tests."""

from .api_client import (
    CollectorAPIClient,
    RawAPIClient,
    generate_random_embedding,
    generate_deterministic_embedding,
    create_test_span,
    create_test_log,
    create_invalid_embedding,
    create_edge_case_prompt,
)
from .chain_utils import (
    verify_registration_onchain,
    verify_weights_onchain,
    wait_for_blocks,
    get_block_with_retry,
    get_neuron_info,
    wait_for_registration,
)
from .testnet_funding import (
    request_faucet_funds,
    transfer_tao,
    wait_for_balance,
    get_balance,
    estimate_registration_cost,
    fund_wallet_for_registration,
)
from .wallet_utils import (
    create_test_wallet,
    cleanup_test_wallet,
    cleanup_all_test_wallets,
    generate_unique_wallet_name,
    get_test_wallet_count,
    get_validator_wallet,
    verify_wallet_exists,
    sign_message,
    create_signed_headers,
    get_hotkey_address,
)

__all__ = [
    # API client
    "CollectorAPIClient",
    "RawAPIClient",
    "generate_random_embedding",
    "generate_deterministic_embedding",
    "create_test_span",
    "create_test_log",
    "create_invalid_embedding",
    "create_edge_case_prompt",
    # Chain utils
    "verify_registration_onchain",
    "verify_weights_onchain",
    "wait_for_blocks",
    "get_block_with_retry",
    "get_neuron_info",
    "wait_for_registration",
    # Testnet funding
    "request_faucet_funds",
    "transfer_tao",
    "wait_for_balance",
    "get_balance",
    "estimate_registration_cost",
    "fund_wallet_for_registration",
    # Wallet utils
    "create_test_wallet",
    "cleanup_test_wallet",
    "cleanup_all_test_wallets",
    "generate_unique_wallet_name",
    "get_test_wallet_count",
    "get_validator_wallet",
    "verify_wallet_exists",
    "sign_message",
    "create_signed_headers",
    "get_hotkey_address",
]
