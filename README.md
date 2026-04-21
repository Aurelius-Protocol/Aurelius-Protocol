# Aurelius Protocol

A Bittensor subnet for moral reasoning alignment. Miners submit structured ethical-dilemma
scenario configurations; validators score them through an 8-stage pipeline and run
accepted scenarios through [Concordia](https://github.com/google-deepmind/concordia)
generative-agent simulations. The resulting transcripts form training data that improves
LLM performance on moral reasoning benchmarks (MoReBench).

---

## 🚨 Run the published Docker image

The canonical way to operate a validator or miner is to pull the image we publish to
**public GHCR** — no registry auth required:

| | Testnet (subnet **455**, `test`) | Mainnet (subnet **37**, `finney`) |
|---|---|---|
| Validator image | `ghcr.io/aurelius-protocol/aurelius-validator:testnet` | `ghcr.io/aurelius-protocol/aurelius-validator:latest` |
| Miner image | `ghcr.io/aurelius-protocol/aurelius-miner:testnet` | `ghcr.io/aurelius-protocol/aurelius-miner:latest` |
| Simulation sidecar | `…/aurelius-concordia:testnet` (pulled automatically) | `…/aurelius-concordia:latest` |

**Do not run the validator from source in production.** The validator's stage-7 pipeline
spawns sandboxed Concordia simulation containers via a Docker socket; running from source
means that stage fails, simulations don't execute, submissions get stuck, and the
validator does not score miners correctly. Source checkouts are supported for development
and CI only — see [Development](#development).

Every push to `main` (mainnet) or `testnet` (testnet) rebuilds the images.

---

## Quickstart — Mainnet validator (SN 37)

Prerequisites:

- Docker 20.10+ and `docker compose`
- A Bittensor wallet **registered on mainnet `netuid 37`**
  (`btcli subnet register --netuid 37 --network finney`) — registration costs TAO
- An OpenAI-compatible LLM API key — [DeepSeek](https://platform.deepseek.com/) is the
  default and cheapest; OpenAI / Anthropic also work

### 1. Write a **minimal** `.env`

This is the entire operator-side config. Set only these four lines; delete anything else
(old `CENTRAL_API_URL`, `BT_NETUID`, `BT_SUBTENSOR_NETWORK`, `TESTLAB_MODE`, etc.) from any
prior `.env` you had. Those are auto-configured by the `ENVIRONMENT` profile baked into
the image (see [`aurelius/config.py`](aurelius/config.py)), and leaving stale values —
especially empty ones like `CENTRAL_API_URL=` — silently breaks config resolution.

```bash
cat > .env <<'EOF'
ENVIRONMENT=mainnet
WALLET_NAME=<your-wallet>
WALLET_HOTKEY=<your-hotkey>
LLM_API_KEY=<your-openai-compatible-api-key>
EOF
```

That's it. Don't add more variables unless you have a specific reason to override a
profile default.

### 2. Docker compose with socket-proxy sidecar

The validator needs Docker daemon access to spawn simulation containers; we gate that
through [`tecnativa/docker-socket-proxy`](https://github.com/Tecnativa/docker-socket-proxy)
so a compromise of the validator container cannot drive the host's daemon directly.

```bash
cat > docker-compose.yml <<'EOF'
services:
  aurelius-validator:
    image: ghcr.io/aurelius-protocol/aurelius-validator:latest
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
    labels:
      com.centurylinklabs.watchtower.enable: "true"

  docker-proxy:
    image: tecnativa/docker-socket-proxy:0.3.0
    container_name: docker-proxy
    restart: unless-stopped
    environment: { CONTAINERS: 1, IMAGES: 1, POST: 1, NETWORKS: 1 }
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
EOF
```

### 3. Bring it up

```bash
mkdir -p data simdata
docker compose up -d
docker compose logs -f aurelius-validator
```

The first minute of logs should show: `Validator permit confirmed`, `Authenticated with
Central API`, `Clock drift check passed`, `Remote config refreshed`, `Config summary |
env=mainnet network=finney api_url=https://new-collector-api-production.up.railway.app
llm_model=deepseek-chat … burn_mode=True`, then cycle summaries every few minutes.

If any of the first four lines are missing or followed by warnings, see
[Troubleshooting](#troubleshooting).

---

## Quickstart — Testnet validator (SN 455)

Identical shape, different tag and environment:

```bash
cat > .env <<'EOF'
ENVIRONMENT=testnet
WALLET_NAME=<your-wallet>
WALLET_HOTKEY=<your-hotkey>
LLM_API_KEY=<your-openai-compatible-api-key>
EOF
```

Register on `netuid 455` on `network test`, and swap `:latest` → `:testnet` everywhere in
the compose above. Everything else — socket proxy, volumes, healthchecks — is identical.

---

## Quickstart — Miner

Miners don't run simulations and don't need the Docker-socket proxy, so the single-command
form is fine.

```bash
docker pull ghcr.io/aurelius-protocol/aurelius-miner:latest   # :testnet for testnet

cat > .env <<'EOF'
ENVIRONMENT=mainnet
WALLET_NAME=<your-wallet>
WALLET_HOTKEY=<your-hotkey>
AXON_EXTERNAL_IP=<your-public-ip>
AXON_EXTERNAL_PORT=8091
EOF

mkdir -p data
docker run -d \
  --name aurelius-miner \
  --restart unless-stopped \
  --env-file .env \
  -p 8091:8091 \
  -v ~/.bittensor/wallets:/home/appuser/.bittensor/wallets:ro \
  -v "$(pwd)/data:/app/data" \
  ghcr.io/aurelius-protocol/aurelius-miner:latest

docker logs -f aurelius-miner
```

The axon must be reachable on the public IP and port you advertise — otherwise validators
can't query you and you'll earn no emissions.

---

## Auto-update via Watchtower (optional add-on)

Add this to the compose file alongside `aurelius-validator` to auto-pull new images as
we publish them (every 5 min poll):

```yaml
  watchtower:
    image: containrrr/watchtower
    container_name: watchtower
    restart: unless-stopped
    environment:
      DOCKER_API_VERSION: "1.40"
      WATCHTOWER_CLEANUP: "true"
      WATCHTOWER_POLL_INTERVAL: "300"
      WATCHTOWER_LABEL_ENABLE: "true"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

The `aurelius-validator` block in the quickstart already has the
`com.centurylinklabs.watchtower.enable` label that opts into management.

---

## Troubleshooting

### `Failed to authenticate with Central API: All connection attempts failed`

Your validator cannot reach the Central API. Look at the `Config summary` line directly
below this warning — the `api_url=` field tells you where it's pointing.

| `api_url` shows | Diagnosis | Fix |
|---|---|---|
| `http://localhost:8000` | You have `ENVIRONMENT=local` in `.env` | Change to `ENVIRONMENT=mainnet` or `ENVIRONMENT=testnet` |
| `…-staging.up.railway.app` | You have `ENVIRONMENT=testnet` but want mainnet | Change to `ENVIRONMENT=mainnet` |
| Empty, or your own URL | You have an explicit `CENTRAL_API_URL=` in `.env` | **Delete** that line — do not leave `CENTRAL_API_URL=` empty; the profile default only kicks in when the variable is absent |
| The production URL | Network egress is blocked (firewall, DNS, proxy) | Check with `curl -v https://new-collector-api-production.up.railway.app/health` from inside the container |

**Empty-string pitfall**: `CENTRAL_API_URL=` (no value) is *not* the same as omitting the
line. The Python config layer treats empty strings as "use profile default" (since the
post-2026-04-21 image), but older validator images see empty-string as an explicit
override to blank. Always delete, never leave blank.

### `Config summary | env=local network=test …` when you wanted mainnet

Same root cause: `ENVIRONMENT=local` in your `.env`. The local profile sets
`api_url=http://localhost:8000`, `network=test`, `testlab=True` — none of which are what
you want on mainnet. Explicit `BT_NETUID=37` in `.env` won't save you; it just means
you'll query SN 37 neurons while trying to hit a nonexistent local API.

### `Failed to set weights: Hotkey … not registered in subnet 37`

Your hotkey isn't registered on the subnet. Register it:
`btcli subnet register --netuid 37 --network finney --wallet.name <name> --wallet.hotkey <name>`
(costs TAO).

### `Failed to persist ramp-up anchor … Permission denied: '/app/data/…'`

The `./data` bind-mount on the host is owned by a uid other than `1000` (the container's
`appuser`). Fix with `chown -R 1000:1000 ./data ./simdata` on the host.

### Dep conflicts when running from source

**Don't run from source for production** — see the warning at the top. But if you're
developing and hit one of these:

- `RuntimeError: Conflict detected: 'scalecodec' … conflicts with 'cyscale'` — your pip
  resolved `async-substrate-interface>=2`. Downgrade: `pip install 'async-substrate-interface>=1.6,<2'`.
- `ImportError: cannot import name 'ScaleObj' from 'async_substrate_interface.types'` —
  your bittensor is newer than the lock file expects. Use `pip install -r requirements.lock`
  rather than `pip install -e .` so you get the tested combination
  (`bittensor==10.2.0`, `async-substrate-interface==1.6.3`).

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

The `ENVIRONMENT` profile (`local` / `testnet` / `mainnet`) auto-configures subnet,
network, Central API URL, simulation resources, and safety flags. **Operators should not
set any of these variables manually** — the profile default is correct.

These four variables are the entire local-tier surface:

| Variable | Required for | Purpose | Default |
|---|---|---|---|
| `ENVIRONMENT` | both | `local` \| `testnet` \| `mainnet` | `local` |
| `WALLET_NAME` | both | Bittensor coldkey wallet name | `default` |
| `WALLET_HOTKEY` | both | Bittensor hotkey name | `default` |
| `LLM_API_KEY` | validator | OpenAI-compatible LLM key for Concordia | (empty) |

Miner-only additions:

| Variable | Purpose | Default |
|---|---|---|
| `AXON_EXTERNAL_IP` | Public IP the miner advertises | (empty → use local IP) |
| `AXON_EXTERNAL_PORT` | Public port the miner advertises | `8091` |

Optional overrides — set only if you know you need a non-default value:

| Variable | Default | When to set |
|---|---|---|
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | Using a non-DeepSeek LLM |
| `LLM_MODEL` | `deepseek-chat` | Using a non-default model |

See [`aurelius/config.py`](aurelius/config.py) for the authoritative per-profile defaults
and every other knob (simulation tuning, timeouts, queue sizes, etc.).

**Two-tier config model.** The list above is the *local* tier — wallet, network, secrets,
set once at startup. A *remote* tier (polling interval, classifier threshold, novelty
threshold, rate limits, minimum protocol versions) is fetched from the Central API at
runtime, cached for 5 minutes, and refreshed transparently. Operators never set remote
values.

---

## Development

**Only for development or CI.** See the big warning at the top — running from source for
production breaks the simulation stage.

```bash
git clone https://github.com/Aurelius-Protocol/Aurelius-Protocol.git
cd Aurelius-Protocol

python3 -m venv .venv
source .venv/bin/activate

# Use the lock file — it pins the exact tested bittensor + async-substrate-interface
# combination. A loose `pip install -e .` will pick today's latest deps and may land
# in one of the conflict states documented in Troubleshooting.
pip install -r requirements.lock
pip install -e ".[ml,simulation,dev]"

cp .env.example .env
$EDITOR .env                     # ENVIRONMENT=local for a testlab loop

aurelius-validator               # or: aurelius-miner
```

### Tests

```bash
# Fast — no network, no Docker
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
- **Socket proxy.** The Quickstart uses `tecnativa/docker-socket-proxy` so the validator
  container can only invoke the Docker API calls it actually needs (`CONTAINERS`,
  `IMAGES`, `NETWORKS`, `POST`). A raw `/var/run/docker.sock` mount would give the
  validator container full host control — do not use it for long-running deployments.
- **Image digest pinning.** `REQUIRE_IMAGE_DIGEST=1` is on by default in testnet and
  mainnet profiles. The Concordia image digest is auto-pinned by CI after each build,
  so operators don't need to configure `CONCORDIA_IMAGE_DIGEST` themselves — just keep
  the validator image current.
- **Work-token accounting.** Balance is checked in stage 3 but deducted only in stage 8
  after successful simulation. Fail-closed: if the Central API is unreachable during
  balance check, submissions are rejected rather than admitted for free.

---

## Links

- [Bittensor docs](https://docs.bittensor.com)
- [Subnet 455 on taostats (testnet)](https://taostats.io/subnet/455/)
- [Subnet 37 on taostats (mainnet)](https://taostats.io/subnet/37/)
- [GHCR packages](https://github.com/orgs/Aurelius-Protocol/packages)
- [Issues](https://github.com/Aurelius-Protocol/Aurelius-Protocol/issues)

## License

MIT
