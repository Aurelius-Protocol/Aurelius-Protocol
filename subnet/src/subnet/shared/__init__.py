# Miner dependency subset extracted from Aurelius v5 (02375403).
"""subnet.shared — canonicalization, identities, envelope, and signing (App F / §3).

CONTRACT (capsule header, plan §13)
- Guarantees: this package is DEPENDENCY-FREE in-repo — it imports nothing else in this
  repository (third-party crypto bindings are allowed), so both the miner and validator (and,
  via a parallel Elixir port, central-api) can depend on it without cycles. CI-enforced
  (contract `shared-independent`).
- Owns (§12): 37 (App-F byte-exact canonicalization), and jointly 2 (cost_class locally
  derivable) + 31 (chain-derived fanout_window) — the Python side; the Elixir port in
  `CentralApi.Shared` must agree byte-for-byte on the golden vectors in `fixtures/`.
- Why: `content_hash` is the single semantic identity that dedup, novelty, charge, and windowing
  all key on. Two independent implementations that disagree on identity break all four, so this
  code is small, pure, and golden-vectored.
"""
