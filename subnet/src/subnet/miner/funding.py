"""Separate coldkey-only SN37 funding, exact call authority and durable no-resubmit state.

WT credit and retention arithmetic belong exclusively to Central. A chain inclusion,
SDK success or unrelated balance increase never proves that a purchase was credited.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct

from subnet.bittensor_v11 import FinalizedRpc, load_wallet, open_subtensor, require_bittensor_v11
from subnet.miner.funding_client import CHAIN_ID, TREASURY, read_funding, verified_deposit
from subnet.miner.funding_runtime import RUNTIME_PROFILES
from subnet.miner.trust import config_signing_pubkey
from subnet.operator.prepared_extrinsic import (
    decoded_scan_candidate,
    validate_unsigned_dict,
    validate_era_chain_anchor,
)
from subnet.shared.canonicalization import canonical_bytes, parse_canonical_json
from subnet.shared.signing import verify
from subnet.shared.sn37_economics import GENESIS_HASH
from subnet.shared.ss58 import ss58_decode

MAX_U64 = 2**64 - 1


class FundingRefused(RuntimeError):
    """No authority to sign, resubmit, or claim funding success."""


def require(condition, reason):
    if not condition:
        raise FundingRefused(reason)


def amount_rao(value):
    require(
        type(value) is str and re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,9})?", value),
        "alpha amount must be a positive decimal with at most nine decimal places",
    )
    whole, _, fraction = value.partition(".")
    require(len(whole) <= 20, "alpha amount exceeds native u64")
    amount = int(whole) * 1_000_000_000 + int(fraction.ljust(9, "0"))
    require(0 < amount < MAX_U64, "alpha amount is zero, exceeds u64, or selects move-all")
    return amount


def transfer_call(hotkey, amount):
    require(type(amount) is int and 0 < amount < MAX_U64, "invalid alpha amount or move-all")
    # Reviewed475/477/479/481 metadata: SubtensorModule(7).transfer_stake(86), AccountId32 x2,
    # NetUid u16 x2 and AlphaBalance u64. No wrapper, proxy, encrypted call or move-all.
    return (
        bytes.fromhex("0756")
        + ss58_decode(TREASURY)
        + ss58_decode(hotkey)
        + struct.pack("<HHQ", 37, 37, amount)
    )


class FundingJournal:
    """Immutable fsynced records. One lock covers prepare, sign and claim-before-broadcast."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = self.directory.lstat()
        require(
            stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid(),
            "funding journal must be an owned real directory",
        )
        require(info.st_mode & 0o022 == 0, "funding journal must not be writable by other users")

    @contextmanager
    def lock(self):
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def read(self, name):
        require(
            name in {"prepared", "signature", "submission", "finalized", "credited"},
            "invalid journal record",
        )
        path = self.directory / (name + ".json")
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise FundingRefused("unsafe funding journal record") from exc
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            require(
                stat.S_ISREG(info.st_mode)
                and info.st_nlink == 1
                and info.st_uid == os.geteuid()
                and stat.S_IMODE(info.st_mode) == 0o600
                and info.st_size <= 1_048_576,
                "unsafe funding journal record",
            )
            return parse_canonical_json(stream.read())

    def claim(self, name, document):
        existing = self.read(name)
        if existing is not None:
            require(
                canonical_bytes(existing) == canonical_bytes(document),
                "funding intent changed; retained state requires observation",
            )
            return False
        path = self.directory / (name + ".json")
        encoded = json.dumps(
            document, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
        require(len(encoded) <= 1_048_576, "oversized funding record")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True


def chain_scope(rpc):
    snapshot = rpc.snapshot()
    require(
        snapshot.genesis_hash == GENESIS_HASH
        and type(snapshot.runtime_spec_version) is int
        and snapshot.runtime_spec_version in RUNTIME_PROFILES
        and type(snapshot.transaction_version) is int
        and snapshot.transaction_version == 1
        and snapshot.chain == "Bittensor",
        "unreviewed funding chain/runtime",
    )
    metadata_sha256, code_hash = RUNTIME_PROFILES[snapshot.runtime_spec_version]
    require(
        rpc.call("state_getStorageHash", ["0x3a636f6465", snapshot.head_hash]) == code_hash,
        "runtime code differs",
    )
    metadata = rpc.call("state_getMetadata", [snapshot.head_hash])
    require(
        isinstance(metadata, str)
        and metadata.startswith("0x")
        and hashlib.sha256(bytes.fromhex(metadata[2:])).hexdigest() == metadata_sha256,
        "runtime metadata differs",
    )
    return snapshot


def retained_runtime(unsigned):
    version = unsigned.get("spec_version") if type(unsigned) is dict else None
    require(
        type(version) is int and version in RUNTIME_PROFILES, "unreviewed retained funding runtime"
    )
    return version


def matched_execution(rpc, match, version):
    """A matched signature must execute under its prepared exact parent and post profile."""
    height, block_hash = match["block_number"], match["block_hash"]
    require(type(height) is int and height > 0, "invalid matched funding height")
    require(rpc.call("chain_getBlockHash", [height]) == block_hash, "funding finality differs")
    header = rpc.call("chain_getHeader", [block_hash])
    parent = rpc.call("chain_getBlockHash", [height - 1])
    require(
        type(header) is dict
        and isinstance(header.get("number"), str)
        and re.fullmatch(r"0x[0-9a-fA-F]+", header["number"]) is not None
        and int(header["number"], 16) == height
        and header.get("parentHash") == parent,
        "funding execution parent differs",
    )
    metadata_sha256, code_hash = RUNTIME_PROFILES[version]
    for digest in (parent, block_hash):
        runtime = rpc.call("state_getRuntimeVersion", [digest])
        require(
            type(runtime) is dict
            and type(runtime.get("specVersion")) is int
            and runtime["specVersion"] == version
            and type(runtime.get("transactionVersion")) is int
            and runtime["transactionVersion"] == 1,
            "matched funding runtime differs",
        )
        require(
            rpc.call("state_getStorageHash", ["0x3a636f6465", digest]) == code_hash,
            "matched funding runtime code differs",
        )
        metadata = rpc.call("state_getMetadata", [digest])
        require(
            isinstance(metadata, str)
            and metadata.startswith("0x")
            and hashlib.sha256(bytes.fromhex(metadata[2:])).hexdigest() == metadata_sha256,
            "matched funding runtime metadata differs",
        )


def transfer_event_index(events, prepared, extrinsic_index):
    """Require native delivered alpha, destination and a capped TAO fee at exact phase."""
    require(type(events) is list, "invalid native events")
    relevant = []
    for ordinal, record in enumerate(events):
        require(type(record) is dict, "invalid native event record")
        if record.get("phase") == "ApplyExtrinsic":
            require(type(record.get("extrinsic_idx")) is int, "invalid native extrinsic phase")
            if record["extrinsic_idx"] == extrinsic_index:
                relevant.append((ordinal, record["event"]))
        else:
            require(record.get("extrinsic_idx") is None, "unexpected native phase index")

    def one(module, name):
        rows = [
            (index, event)
            for index, event in relevant
            if event.get("module_id") == module and event.get("event_id") == name
        ]
        require(len(rows) == 1, "missing unique native " + name)
        return rows[0]

    success_index, _ = one("System", "ExtrinsicSuccess")
    require(
        not any(e.get("event_id") == "ExtrinsicFailed" for _, e in relevant),
        "native transfer failed",
    )
    transfer_index, transfer = one("SubtensorModule", "StakeTransferred")
    values = transfer["attributes"]
    require(
        type(values) in (list, tuple) and len(values) == 6, "invalid native transfer attributes"
    )
    source, target, hotkey, origin, destination, tao = values
    require(
        (source, target, hotkey) == (prepared["coldkey"], TREASURY, prepared["hotkey"])
        and type(origin) is int
        and type(destination) is int
        and origin == destination == 37
        and type(tao) is int
        and tao > 0,
        "native transfer scope differs",
    )
    movements = []
    for name, account in (("StakeRemoved", source), ("StakeAdded", target)):
        index, event = one("SubtensorModule", name)
        attributes = event["attributes"]
        require(
            type(attributes) in (list, tuple) and len(attributes) == 6,
            "invalid native alpha movement",
        )
        actor, event_hotkey, moved_tao, alpha, netuid, fee = attributes
        require(
            (actor, event_hotkey) == (account, hotkey)
            and all(type(n) is int and n >= 0 for n in (moved_tao, alpha, netuid, fee))
            and (moved_tao, alpha, netuid) == (tao, prepared["alpha_rao"], 37),
            "native delivered alpha differs from intent",
        )
        movements.append(index)
    require(
        movements[0] < movements[1] < transfer_index < success_index,
        "native transfer event ordering differs",
    )
    _, fee_event = one("TransactionPayment", "TransactionFeePaid")
    fee = fee_event["attributes"]
    require(
        type(fee) is dict
        and fee.get("who") == source
        and type(fee.get("actual_fee")) is int
        and 0 <= fee["actual_fee"] <= prepared["fee_cap_tao_rao"]
        and type(fee.get("tip")) is int
        and fee["tip"] == 0,
        "native TAO fee differs from cap",
    )
    return transfer_index


def observe(client, rpc, journal, response):
    """Read an exact prepared signature at a finalized canonical block; never retry a transfer."""
    prepared = journal.read("prepared")
    submission = journal.read("submission")
    if not submission:
        return {"status": "prepared_not_submitted", "executed": False}
    require(
        prepared is not None
        and submission["prepared_sha256"] == hashlib.sha256(canonical_bytes(prepared)).hexdigest(),
        "submission authority differs",
    )
    require(
        prepared["coldkey"] == response["identity"]["coldkey"]
        and prepared["hotkey"] == response["identity"]["hotkey"]
        and prepared["netuid"] == 37,
        "retained funding beneficiary differs",
    )
    authority = validate_unsigned_dict(
        prepared["unsigned"],
        expected_address=prepared["coldkey"],
        expected_genesis_hash=GENESIS_HASH,
        expected_spec_version=retained_runtime(prepared["unsigned"]),
        expected_transaction_version=1,
        expected_period=64,
        expected_tip=0,
        expected_call_data_sha256=hashlib.sha256(
            transfer_call(prepared["hotkey"], prepared["alpha_rao"])
        ).hexdigest(),
    )
    signature = bytes.fromhex(journal.read("signature")["signature_hex"])
    require(
        verify(signature, authority.payload, ss58_decode(prepared["coldkey"])),
        "retained funding signature differs",
    )
    finalized = journal.read("finalized")
    if finalized:
        retained_credit = journal.read("credited")
        receipt_source = response
        if retained_credit is not None:
            require(
                retained_credit["deposit_key"] == finalized["deposit_key"],
                "retained credit key differs",
            )
            receipt_source = {
                **response,
                "account": {"receipts": {"real": [retained_credit["row"]]}},
            }
        receipt = verified_deposit(receipt_source, finalized["deposit_key"])
        if receipt is not None:
            require(
                receipt["alpha_amount_rao"] == prepared["alpha_rao"]
                and receipt["price_policy_digest"] == prepared["price_policy_digest"],
                "credited amount or price differs from intent",
            )
            row = next(
                row
                for row in receipt_source["account"]["receipts"]["real"]
                if row["receipt"] == receipt
            )
            journal.claim("credited", {"row": row, "deposit_key": finalized["deposit_key"]})
            return {"status": "credited", "receipt": receipt, "executed": False}
        return {
            "status": "awaiting_wt_credit",
            "deposit_key": finalized["deposit_key"],
            "executed": False,
        }
    snapshot = chain_scope(rpc)
    matches = []
    for height in range(
        authority.scan_start_block, min(snapshot.block_number, authority.scan_end_block) + 1
    ):
        block_hash = rpc.call("chain_getBlockHash", [height])
        block = rpc.call("chain_getBlock", [block_hash])
        for index, raw in enumerate(block["block"]["extrinsics"]):
            decoded = client.decode_scale("Extrinsic", bytes.fromhex(raw[2:]))
            candidate = decoded_scan_candidate(
                decoded,
                raw,
                block_hash=block_hash,
                block_number=height,
                extrinsic_index=index,
                authority=authority,
                signature=signature,
            )
            if candidate:
                matches.append(candidate)
    require(len(matches) <= 1, "multiple finalized matches for the funding signature")
    if not matches:
        return {
            "status": "submission_unresolved",
            "executed": False,
            "reason": "observe_same_intent; never automatically resubmit",
        }
    match = matches[0]
    matched_execution(rpc, match, authority.spec_version)
    snapshot_at = client.at(match["block_number"])
    events = snapshot_at.query(("System", "Events"), [])
    if hasattr(events, "value"):
        events = events.value
    require(
        snapshot_at.block_info().hash == match["block_hash"],
        "SDK events anchor differs from canonical finality",
    )
    event_index = transfer_event_index(events, prepared, match["extrinsic_index"])
    deposit_key = f"{CHAIN_ID}‖{match['block_hash']}‖{match['extrinsic_index']}‖{event_index}"
    journal.claim("finalized", {**match, "deposit_key": deposit_key})
    return {"status": "awaiting_wt_credit", "deposit_key": deposit_key, "executed": False}


def fund(client, wallet, *, bt, rpc, response, alpha_rao, max_fee_rao, execute, journal):
    require(
        type(max_fee_rao) is int and 0 <= max_fee_rao <= 100_000_000, "fee cap must be 0..0.1 TAO"
    )
    coldkey = wallet.coldkeypub.ss58_address
    hotkey = response["identity"]["hotkey"]
    require(
        response["identity"] == {"coldkey": coldkey, "hotkey": hotkey, "netuid": 37},
        "funding wallet must own the serving hotkey",
    )
    with journal.lock():
        if journal.read("submission"):
            return observe(client, rpc, journal, response)
        require(
            response["terms"]["status"] == "available",
            "funding unavailable: " + response["terms"].get("reason", "unknown"),
        )
        snapshot = chain_scope(rpc)
        require(
            snapshot.runtime_spec_version == response["terms"]["runtime_spec_version"],
            "funding terms runtime differs from finalized chain",
        )
        require(
            0 <= snapshot.block_number - response["finalized_block"] <= 64,
            "funding terms are ahead of chain or stale",
        )
        read = client.at(snapshot.block_number)
        require(
            read.query(("SubtensorModule", "Owner"), [hotkey]) == coldkey,
            "source hotkey ownership differs",
        )
        require(
            type(read.query(("SubtensorModule", "Uids"), [37, hotkey])) is int,
            "source hotkey is not registered on subnet 37",
        )
        require(
            read.query(("SubtensorModule", "TransferToggle"), [37]) is True
            and read.query(("SubtensorModule", "SubtokenEnabled"), [37]) is True,
            "subnet stake transfers unavailable",
        )
        position = read.staking.get(coldkey, hotkey, 37)
        collateral = read.query(("SubtensorModule", "MinerCollateral"), [37, hotkey, coldkey])
        locked = 0 if collateral is None else collateral["locked"]
        availability = read.runtime(
            bt.runtime_api.StakeInfoRuntimeApi.get_stake_availability_for_coldkeys,
            [[coldkey], [37]],
        )
        available = availability[coldkey][37]["available"]
        require(
            type(position.rao) is int
            and type(locked) is int
            and type(available) is int
            and 0 < alpha_rao <= min(position.rao - locked, available),
            "insufficient available alpha after locks/collateral",
        )
        # This is a chain transfer check, not WT pricing. The pinned runtime's
        # DefaultMinTransfer is 100,000 TAO-rao. Its API truncates the Q64 price
        # to nano-TAO, so this integer product is a conservative lower bound.
        price = read.runtime(bt.runtime_api.SwapRuntimeApi.current_alpha_price, [37])
        require(
            type(price) is int and price > 0 and alpha_rao * price // 1_000_000_000 >= 100_000,
            "alpha amount is below the native minimum transfer value (0.0001 TAO)",
        )
        free = read.balances.get(coldkey)
        require(type(free.rao) is int and free.rao >= 0, "invalid free TAO balance")
        # The reviewed native fee handler prefers TAO and falls back to free alpha
        # for transfer_stake. The cap bounds the estimated fee; it is not a TAO
        # reserve requirement. Runtime validation selects and checks the fee source.
        intent = bt.TransferStake(
            dest_coldkey_ss58=TREASURY,
            hotkey_ss58=hotkey,
            origin_netuid=37,
            dest_netuid=37,
            amount_alpha=bt.Balance.from_rao(alpha_rao, netuid=37),
        )
        policy = bt.Policy(
            max_fee_tao=bt.Balance.from_rao(max_fee_rao),
            allowed_netuids=[37],
            allow_raw_calls=False,
        )
        plan = client.plan(intent, wallet, policy=policy)
        require(
            plan.ok
            and plan.op == "transfer_stake"
            and plan.signer == "coldkey"
            and plan.signer_address == coldkey
            and plan.fee is not None
            and plan.fee.netuid == 0
            and type(plan.fee.rao) is int
            and 0 <= plan.fee.rao <= max_fee_rao
            and plan.call.spec_version == snapshot.runtime_spec_version
            and plan.call.data == transfer_call(hotkey, alpha_rao),
            "planned transfer/fee differs from exact authority",
        )
        preview = {
            "status": "preview",
            "executed": False,
            "coldkey": coldkey,
            "hotkey": hotkey,
            "netuid": 37,
            "destination": TREASURY,
            "alpha_rao": alpha_rao,
            "fee_cap_tao_rao": max_fee_rao,
            "planned_fee_tao_rao": plan.fee.rao,
            "free_tao_rao": free.rao,
            "fee_payment": "chain_selected_tao_or_alpha",
            "price_policy_digest": response["terms"]["price_policy_digest"],
        }
        if not execute:
            return preview
        prepared = journal.read("prepared")
        if prepared is None:
            unsigned = client.prepare_call(
                plan.call, address=coldkey, crypto_type=1, period=64, tip=0
            )
        else:
            require(
                prepared["coldkey"] == coldkey
                and prepared["hotkey"] == hotkey
                and prepared["alpha_rao"] == alpha_rao
                and prepared["fee_cap_tao_rao"] == max_fee_rao
                and prepared["price_policy_digest"] == preview["price_policy_digest"],
                "funding intent changed; retained state requires observation",
            )
            unsigned = bt.UnsignedExtrinsic.from_dict(prepared["unsigned"])
        authority = validate_unsigned_dict(
            unsigned.to_dict(),
            expected_address=coldkey,
            expected_genesis_hash=GENESIS_HASH,
            expected_spec_version=snapshot.runtime_spec_version,
            expected_transaction_version=1,
            expected_period=64,
            expected_tip=0,
            expected_call_data_sha256=hashlib.sha256(transfer_call(hotkey, alpha_rao)).hexdigest(),
        )
        latest = chain_scope(rpc)
        require(
            authority.spec_version == latest.runtime_spec_version
            and authority.era_current <= latest.block_number <= authority.scan_end_block,
            "prepared transfer is ahead of finality or expired",
        )
        validate_era_chain_anchor(
            authority,
            current_finalized_block_number=authority.era_current,
            birth_block_hash=rpc.call("chain_getBlockHash", [authority.era_birth]),
        )
        if prepared is None:
            prepared = {
                **preview,
                "unsigned": unsigned.to_dict(),
                "status": "prepared_before_signature",
            }
            journal.claim("prepared", prepared)
        retained_signature = journal.read("signature")
        if retained_signature is None:
            signer = wallet.coldkey
            require(signer.ss58_address == coldkey, "unlocked coldkey differs")
            signature = signer.sign(authority.payload)
        else:
            signature = bytes.fromhex(retained_signature["signature_hex"])
        require(
            type(signature) is bytes
            and len(signature) == 64
            and verify(signature, authority.payload, ss58_decode(coldkey)),
            "invalid coldkey signature",
        )
        journal.claim("signature", {"signature_hex": signature.hex()})
        latest = chain_scope(rpc)
        require(
            latest.runtime_spec_version == authority.spec_version
            and latest.block_number <= authority.scan_end_block,
            "mortal transfer runtime changed or expired before broadcast",
        )
        require(
            journal.claim(
                "submission",
                {
                    "status": "submission_ambiguous_until_observed",
                    "prepared_sha256": hashlib.sha256(canonical_bytes(prepared)).hexdigest(),
                },
            ),
            "submission already claimed",
        )
        # The claim is permanent even if transport, finality wait, process or disk fails next.
        raw_policy = bt.Policy(
            max_fee_tao=bt.Balance.from_rao(max_fee_rao), allowed_netuids=[37], allow_raw_calls=True
        )
        try:
            client.submit_signature(
                unsigned,
                signature,
                policy=raw_policy,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
        except Exception as exc:
            raise FundingRefused(
                "submission ambiguous; run the same intent again to observe, never send a replacement"
            ) from exc
        return observe(client, rpc, journal, response)


def run(
    *, base_url, endpoint, wallet_path, wallet_name, hotkey_name, alpha, fee_cap, execute, directory
):
    from subnet.miner.serve import minter_from_env

    minter = minter_from_env(os.environ.get("MINER_HOTKEY", "default"))
    response = read_funding(
        base_url,
        hotkey=minter.hotkey_ss58,
        signer=minter.sign_economics,
        config_pubkey=config_signing_pubkey(),
    )
    bt = require_bittensor_v11()
    wallet = load_wallet(bt, name=wallet_name, hotkey=hotkey_name, path=wallet_path)
    rpc = FinalizedRpc(endpoint)
    try:
        with open_subtensor(bt, endpoint) as client:
            return fund(
                client,
                wallet,
                bt=bt,
                rpc=rpc,
                response=response,
                alpha_rao=amount_rao(alpha),
                max_fee_rao=amount_rao(fee_cap),
                execute=execute,
                journal=FundingJournal(directory),
            )
    finally:
        rpc.close()
