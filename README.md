# Aurelius Protocol

A Bittensor subnet for moral reasoning alignment. Miners submit structured ethical-dilemma
scenario configurations; validators score them through an 8-stage pipeline and run
accepted scenarios through [Concordia](https://github.com/google-deepmind/concordia)
generative-agent simulations. The resulting transcripts form training data that improves
LLM performance on moral reasoning benchmarks (MoReBench).

Every push to `main` or `testnet` publishes container images to public GHCR.
No registry auth is required to pull.

| | Testnet (subnet 455, `test` network) | Mainnet (subnet 37, `finney`) |
|---|---|---|
| Validator image | `ghcr.io/aurelius-protocol/aurelius-validator:testnet` | `ghcr.io/aurelius-protocol/aurelius-validator:latest` |
| Miner image | `ghcr.io/aurelius-protocol/aurelius-miner:testnet` | `ghcr.io/aurelius-protocol/aurelius-miner:latest` |
| Simulation image | `…/aurelius-concordia:testnet` (pulled automatically by validators) | `…/aurelius-concordia:latest` |

The quickstart examples below use the testnet tags. For mainnet, set
`ENVIRONMENT=mainnet` in the `.env` and swap every `:testnet` reference to
`:latest`.

---

## Quickstart — Testnet Validator

Use **Docker Compose with docker-socket-proxy** (below) for anything long-lived —
the validator needs Docker daemon access to spawn sandboxed simulation
containers, and the proxy restricts that to the minimum set of API calls it
actually uses. A raw-socket `docker run` for quick exploration is shown at the
end of this section with appropriate caveats.

Prerequisites:

- Docker 20.10+ and `docker compose`
- A Bittensor wallet registered on testnet `netuid 455`
  (`btcli subnet register --netuid 455 --network test`)
- An OpenAI-compatible LLM API key — [DeepSeek](https://platform.deepseek.com/) is the
  default and cheapest; OpenAI / Anthropic also work

```bash
# 1. Write a minimal .env — ENVIRONMENT=testnet auto-sets netuid, network,
#    Central API URL, simulation resources, and safety flags.
cat > .env <<'EOF'
ENVIRONMENT=testnet
WALLET_NAME=your-wallet
WALLET_HOTKEY=your-hotkey
LLM_API_KEY=sk-...
EOF

# 2. Create a compose file with a docker-socket-proxy sidecar so the
#    validator can launch sim containers without direct daemon access.
cat > docker-compose.yml <<'EOF'
services:
  aurelius-validator:
    image: ghcr.io/aurelius-protocol/aurelius-validator:testnet
    container_name: aurelius-validator
    restart: unless-stopped
    env_file: .env
    environment:
      DOCKER_HOST: tcp://docker-proxy:2375
    cap_add: [NET_ADMIN]
    volumes:
      - ~/.bittensor/wallets:/home/appuser/.bittensor/wallets:ro
      - ./data:/app/data
      - ./simdata:/sim-data
    depends_on: [docker-proxy]

  docker-proxy:
    image: tecnativa/docker-socket-proxy:0.3.0
    container_name: docker-proxy
    restart: unless-stopped
    environment: { CONTAINERS: 1, IMAGES: 1, POST: 1, NETWORKS: 1 }
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
EOF

# 3. Bring it up
mkdir -p data simdata
docker compose up -d
docker compose logs -f aurelius-validator
```

<details>
<summary><b>Raw-socket single-command alternative — local exploration only</b></summary>

A one-container equivalent that mounts `/var/run/docker.sock` directly. **Not
recommended for long-running deployments**: any compromise of the validator
container can drive the host's Docker daemon directly (`docker run
--privileged`, mount arbitrary volumes, etc.). Use only on a disposable VM
where you control nothing else.

```bash
docker pull ghcr.io/aurelius-protocol/aurelius-validator:testnet
cat > .env <<'EOF'
ENVIRONMENT=testnet
WALLET_NAME=your-wallet
WALLET_HOTKEY=your-hotkey
LLM_API_KEY=sk-...
EOF
mkdir -p data simdata
docker run -d \
  --name aurelius-validator \
  --restart unless-stopped \
  --env-file .env \
  -v ~/.bittensor/wallets:/home/appuser/.bittensor/wallets:ro \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/simdata:/sim-data" \
  ghcr.io/aurelius-protocol/aurelius-validator:testnet
```

</details>

---

## Quickstart — Testnet Miner

Prerequisites:

- Docker 20.10+
- A Bittensor wallet registered on testnet `netuid 455`
- A publicly reachable IP and an open inbound TCP port for the axon (default `8091`)

```bash
# 1. Pull the latest testnet image
docker pull ghcr.io/aurelius-protocol/aurelius-miner:testnet

# 2. Write a minimal .env
cat > .env <<'EOF'
ENVIRONMENT=testnet
WALLET_NAME=your-wallet
WALLET_HOTKEY=your-hotkey
AXON_EXTERNAL_IP=<your-public-ip>
AXON_EXTERNAL_PORT=8091
EOF

# 3. Run
mkdir -p data
docker run -d \
  --name aurelius-miner \
  --restart unless-stopped \
  --env-file .env \
  -p 8091:8091 \
  -v ~/.bittensor/wallets:/home/appuser/.bittensor/wallets:ro \
  -v "$(pwd)/data:/app/data" \
  ghcr.io/aurelius-protocol/aurelius-miner:testnet

docker logs -f aurelius-miner
```

---

## How It Works

```
Miner                      Validator                       Central API
  |                            |                                |
  |   ScenarioConfigSynapse    |                                |
  |--------------------------->|                                |
  |   (scenario_config,        |  1. version check              |
  |    work_id, signature)     |  2. schema validation          |
  |                            |  3. work-token balance ------->|
  |                            |  4. rate-limit (per hotkey)    |
  |                            |  5. novelty check (FAISS)      |
  |                            |  6. classifier quality gate    |
  |                            |  7. Concordia simulation       |
  |                            |     (sandboxed container)      |
  |                            |  8. work-token deduct -------->|
  |                            |     + on-chain weight set      |
  |                            |  report submission ----------->|
```

The pipeline short-circuits on the first failure and only deducts the work-token after
all eight stages pass. The Concordia simulation runs in an ephemeral container with
CPU/RAM limits scaled to the scenario's agent count, and its LLM egress is firewalled to
the allowlist in `SIM_ALLOWED_LLM_HOSTS`. Transcripts are parsed, scored for coherence,
and become the payload that determines the miner's on-chain weight.

Code landmarks: wire format in [`aurelius/protocol.py`](aurelius/protocol.py), pipeline
in [`aurelius/validator/pipeline.py`](aurelius/validator/pipeline.py), simulation runner
in [`aurelius/simulation/docker_runner.py`](aurelius/simulation/docker_runner.py).

---

## Configuration

Setting `ENVIRONMENT` selects a profile and auto-configures subnet, network, Central API
URL, simulation resources, and safety flags. The six variables below cover nearly every
operator deployment:

| Variable | Required for | Purpose | Default |
|---|---|---|---|
| `ENVIRONMENT` | both | `local` \| `testnet` \| `mainnet` — selects profile | `local` |
| `WALLET_NAME` | both | Bittensor coldkey wallet name | `default` |
| `WALLET_HOTKEY` | both | Bittensor hotkey name | `default` |
| `LLM_API_KEY` | validator | OpenAI-compatible LLM key for Concordia | (empty) |
| `LLM_BASE_URL` | validator (opt.) | Override LLM endpoint | `https://api.deepseek.com/v1` |
| `LLM_MODEL` | validator (opt.) | Override model name | `deepseek-chat` |
| `AXON_EXTERNAL_IP` | miner | Public IP the miner advertises | (empty → use local IP) |
| `AXON_EXTERNAL_PORT` | miner | Public port the miner advertises | `8091` |

See [`.env.example`](.env.example) for the full surface and
[`aurelius/config.py`](aurelius/config.py) for the authoritative defaults per profile.

**Two-tier config model.** The list above is the *local* tier (wallet, network, secrets —
set at startup, never changes). A *remote* tier (polling interval, classifier threshold,
novelty threshold, rate limits, minimum protocol versions) is fetched from the Central
API at runtime, cached for 5 minutes, and refreshed transparently. Operators do not set
any of the remote-tier values — they live server-side.

---

## Auto-update via Watchtower (optional add-on)

If you want the validator to roll forward automatically whenever a new
`:testnet` (or `:latest` for mainnet) image is published, append a watchtower
service to the compose file from the Quickstart. Watchtower polls GHCR every
5 minutes and only restarts containers that opt in via the
`com.centurylinklabs.watchtower.enable: "true"` label.

```yaml
# append to docker-compose.yml
  watchtower:
    image: containrrr/watchtower
    container_name: watchtower
    restart: unless-stopped
    environment:
      WATCHTOWER_CLEANUP: "true"
      WATCHTOWER_POLL_INTERVAL: "300"
      WATCHTOWER_LABEL_ENABLE: "true"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    labels:
      com.centurylinklabs.watchtower.enable: "true"
```

Then add `labels: { com.centurylinklabs.watchtower.enable: "true" }` to the
`aurelius-validator` service block so watchtower manages it. GHCR packages
under `aurelius-protocol` are public, so no registry credentials are mounted.

---

## Running From Source

For development, or if you prefer not to use Docker:

```bash
git clone https://github.com/Aurelius-Protocol/Aurelius-Protocol.git
cd Aurelius-Protocol

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[ml,simulation]"   # validator + miner runtime deps

cp .env.example .env
$EDITOR .env                         # fill in ENVIRONMENT, wallet, etc.

aurelius-validator                   # or: aurelius-miner
```

Extras: `[ml]` for embeddings/classifier, `[simulation]` for the Concordia Docker SDK,
`[benchmark]` for fine-tuning / MoReBench evaluation, `[dev]` for pytest and ruff.
`aurelius-deposit` is a one-shot CLI for verifying coldkey deposits.

---

## Development & Testing

```bash
pip install -e ".[ml,simulation,dev]"

# Fast tests — no network, no Docker
pytest tests/ --ignore=tests/e2e --ignore=tests/common/test_embeddings.py

# Full suite including Docker-dependent simulation tests
pytest tests/

# E2E (requires a running testnet and a funded wallet)
pytest tests/e2e/ -m e2e

# Lint / format
ruff check aurelius/
ruff format aurelius/
```

---

## Security Notes

- **Concordia isolation.** Every simulation runs in an ephemeral Docker container with
  capped RAM / CPU, egress limited to `SIM_ALLOWED_LLM_HOSTS`, and no persistent
  filesystem outside the mounted `/sim-data`.
- **Socket proxy.** The Quickstart uses `tecnativa/docker-socket-proxy` so the
  validator container can only invoke the Docker API calls it actually needs
  (`CONTAINERS`, `IMAGES`, `NETWORKS`, `POST`). The raw `/var/run/docker.sock`
  mount is only shown as an exploration alternative and is not recommended for
  long-running deployments.
- **Image digest pinning.** `REQUIRE_IMAGE_DIGEST=1` is on by default in the testnet and
  mainnet profiles. The Concordia image digest is auto-pinned by CI after each build,
  so operators don't need to configure `CONCORDIA_IMAGE_DIGEST` themselves — just keep
  your validator image current.
- **Work-token accounting.** Balance is checked in stage 3 but deducted only in stage 8,
  after successful simulation. Fail-closed behavior: if the Central API is unreachable
  during balance check, submissions are rejected rather than admitted for free.

---

## Mainnet

For mainnet, use `ENVIRONMENT=mainnet` in the `.env` and the `:latest` image
tag everywhere the quickstart uses `:testnet`. Everything else — compose
topology, socket proxy, watchtower, `.env` fields — is identical. `:latest`
tracks the `main` branch; the v2→v3 cutover is complete as of the commit
that merged v3 to `main`.

---

## Links

- [Bittensor docs](https://docs.bittensor.com)
- [Subnet 455 on taostats (testnet)](https://taostats.io/subnet/455/)
- [GHCR packages](https://github.com/orgs/Aurelius-Protocol/packages)
- [Issues](https://github.com/Aurelius-Protocol/Aurelius-Protocol/issues)

## License

MIT
