# Approved runtime477 miner compatibility port

Copies the already reviewed v5 miner ServeAxon runtime477 allowlist and exact
SN37 tariff scope support; pricing and trust keys are unchanged. Unknown runtimes
and wrong network scope still refuse. The public README now names the reviewed
477 pin and the adopted 7,200-finalized-block notice. This does not promote the
separate historical execution class or claim that public funding is active.

An isolated GH200 Kubernetes Job installed this public repository with the frozen
lockfile, ran all public miner/shared tests, Ruff and import contracts, built its
wheel, installed production dependencies and the wheel in fresh environments,
and imported all 61 installed modules without private platform packages. The
installed wheel verified the bundled production configuration key automatically.
The result and source/wheel digests are recorded alongside the full log. No wallet,
signer, private validator or live funding transaction was used.
