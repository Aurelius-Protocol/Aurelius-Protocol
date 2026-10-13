"""subnet.miner (STUB, platform) — axon server, staging pool, CLI.

CONTRACT (capsule header)
- Owns (§12): 1 (no miner-side token deduction — the only spend is the server-side terminal
  charge). In staging mode the validator selects only an operator-defined task type/schema; the
  miner selects the sole valid candidate eligible for the request's fanout window and serves it
  with a FRESH `work_id`. Legacy unscoped lanes retain append-order selection until they publish
  window metadata. Nothing about payment/novelty/quality is computed miner-side.
- CLI (v1 scope): `deposit` / `balance` / `validate` / `serve`; coldkey signing is local-only.
  Lane-owned operator tooling, such as `aurelius-sae-stage`, is a separate entry point and never
  expands the generic miner's task-specific imports.
"""
