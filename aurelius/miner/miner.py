"""Miner implementation - submits prompts to validators."""

import argparse
import sys

import bittensor as bt

from aurelius.shared.config import Config, ConfigurationError
from aurelius.shared.protocol import PromptSynapse


def send_prompt(
    prompt: str,
    validator_uid: int = 0,
    vendor: str | None = None,
    model_requested: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    min_chars: int | None = None,
    max_chars: int | None = None,
) -> str:
    """
    Send a prompt to a validator and return the response.

    Args:
        prompt: The prompt text to send
        validator_uid: The UID of the validator to query (default: 0)
        vendor: AI vendor to use (e.g., 'openai', 'anthropic')
        model_requested: Specific model to use (e.g., 'o4-mini', 'gpt-4o')
        temperature: Sampling temperature (0.0-2.0)
        top_p: Nucleus sampling parameter (0.0-1.0)
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
        min_chars: Minimum response length in characters
        max_chars: Maximum response length in characters

    Returns:
        The response from the validator (OpenAI completion)
    """
    # Setup logging
    Config.setup_logging()

    # Apply network-aware defaults based on BT_NETUID
    Config.apply_network_defaults()

    # Detect wallet if not explicitly configured
    try:
        Config.detect_and_set_wallet(role="miner")
    except ConfigurationError as e:
        bt.logging.error(str(e))
        sys.exit(1)

    bt.logging.info(f"Initializing miner with wallet: {Config.MINER_WALLET_NAME}")

    # Initialize wallet
    wallet = bt.Wallet(name=Config.MINER_WALLET_NAME, hotkey=Config.MINER_HOTKEY)

    # Initialize dendrite (for sending requests)
    dendrite = bt.Dendrite(wallet=wallet)

    bt.logging.info(f"Prompt: {prompt}")
    if vendor or model_requested:
        bt.logging.info(f"Model specs: vendor={vendor}, model={model_requested}")
    if temperature is not None:
        bt.logging.info(f"Temperature: {temperature}")

    # Create the synapse with the prompt and model specifications
    synapse = PromptSynapse(
        prompt=prompt,
        vendor=vendor,
        model_requested=model_requested,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        min_chars=min_chars,
        max_chars=max_chars,
    )

    # Determine target axon based on mode
    if Config.LOCAL_MODE:
        # Local mode: Connect directly to validator IP:PORT
        bt.logging.info("=" * 60)
        bt.logging.info("LOCAL MODE ENABLED")
        bt.logging.info("=" * 60)
        bt.logging.info(f"Connecting directly to validator at {Config.VALIDATOR_HOST}:{Config.BT_PORT_VALIDATOR}")

        # Create a custom axon info for the local validator
        target_axon = bt.AxonInfo(
            version=1,
            ip=Config.VALIDATOR_HOST,
            port=Config.BT_PORT_VALIDATOR,
            ip_type=4,
            hotkey="local_validator",  # Dummy hotkey for local mode
            coldkey="local_coldkey",
        )
    else:
        # Normal mode: Query metagraph for validator UID
        subtensor_config = Config.get_subtensor_config()
        subtensor = bt.Subtensor(**subtensor_config)
        metagraph = subtensor.metagraph(netuid=Config.BT_NETUID)

        bt.logging.info(f"Sending prompt to validator UID {validator_uid}")
        target_axon = metagraph.axons[validator_uid]

    # Query the validator
    try:
        responses = dendrite.query(
            axons=target_axon,
            synapse=synapse,
            timeout=30,
            deserialize=False,  # Get full synapse back, not just deserialized string
        )

        # dendrite.query can return different types:
        # - Sometimes the deserialized response (string from synapse.deserialize())
        # - Sometimes the synapse object itself
        # - Sometimes a list

        # Extract the synapse from response
        result_synapse = None

        # dendrite.query returns the synapse itself when querying a single axon
        # For lists, it returns a list of synapses
        if isinstance(responses, PromptSynapse):
            result_synapse = responses
        elif isinstance(responses, list) and len(responses) > 0:
            result_synapse = responses[0]
        elif isinstance(responses, str):
            # Shouldn't happen with single axon query, but handle it
            # Create a dummy synapse with the response
            result_synapse = PromptSynapse(prompt=prompt)
            result_synapse.response = responses
        else:
            result_synapse = responses if responses else synapse

        if result_synapse and hasattr(result_synapse, "response") and result_synapse.response:
            # Display results
            print("\n" + "=" * 60)
            print("RESPONSE FROM VALIDATOR")
            print("=" * 60)
            print(f"Prompt:   {result_synapse.prompt}")
            print(f"Response: {result_synapse.response}")
            print(f"Model:    {result_synapse.model_used or 'unknown'}")

            # Display moderation results if available
            if result_synapse.danger_score is not None:
                print("\n--- Moderation Results ---")
                print(f"Danger Score:  {result_synapse.danger_score:.4f}")
                print(f"Accepted:      {'✓ YES' if result_synapse.accepted else '✗ NO'}")

                if result_synapse.rejection_reason:
                    print(f"Rejection:     {result_synapse.rejection_reason}")

                if result_synapse.category_scores:
                    print("\nTop Category Scores:")
                    sorted_cats = sorted(result_synapse.category_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                    for category, score in sorted_cats:
                        if score > 0.01:
                            print(f"  {category:25s}: {score:.4f}")

            print("=" * 60 + "\n")
            return result_synapse.response
        else:
            bt.logging.error("No response received")
            return ""

    except Exception as e:
        bt.logging.error(f"Error querying validator: {e}")
        import traceback

        traceback.print_exc()
        return ""


def main():
    """Main entry point for the miner."""
    parser = argparse.ArgumentParser(description="Bittensor Miner - Submit prompts to validators")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to send to the validator")
    parser.add_argument("--validator-uid", type=int, default=1, help="The UID of the validator to query (default: 1)")
    parser.add_argument("--netuid", type=int, default=None, help=f"Override the netuid (default: {Config.BT_NETUID})")

    # Model specification arguments
    parser.add_argument("--vendor", type=str, default=None, help="AI vendor to use (e.g., 'openai', 'anthropic')")
    parser.add_argument("--model", type=str, default=None, help="Specific model to use (e.g., 'o4-mini', 'gpt-4o')")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (0.0-2.0)")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling parameter (0.0-1.0)")
    parser.add_argument("--frequency-penalty", type=float, default=None, help="Frequency penalty (-2.0 to 2.0)")
    parser.add_argument("--presence-penalty", type=float, default=None, help="Presence penalty (-2.0 to 2.0)")
    parser.add_argument("--min-chars", type=int, default=None, help="Minimum response length in characters")
    parser.add_argument("--max-chars", type=int, default=None, help="Maximum response length in characters")

    args = parser.parse_args()

    # Override netuid if provided
    if args.netuid is not None:
        Config.BT_NETUID = args.netuid

    # Send the prompt with model specifications
    send_prompt(
        prompt=args.prompt,
        validator_uid=args.validator_uid,
        vendor=args.vendor,
        model_requested=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
