# Aurelius Protocol

A Bittensor subnet for moral reasoning alignment. Miners submit structured ethical-dilemma
scenario configurations; validators score them through an 8-stage pipeline and run
accepted scenarios through [Concordia](https://github.com/google-deepmind/concordia)
generative-agent simulations. The resulting transcripts form training data that improves
LLM performance on moral reasoning benchmarks (MoReBench).

---

## Recommended setup: the published Docker image

We publish validator, miner, and simulation images to **public GHCR** — no registry auth
required:

| | Testnet (subnet **455**, `test`) | Mainnet (subnet **37**, `finney`) |
|---|---|---|
| Validator image | `ghcr.io/aurelius-protocol/aurelius-validator:testnet` | `ghcr.io/aurelius-protocol/aurelius-validator:latest` |
| Miner image | `ghcr.io/aurelius-protocol/aurelius-miner:testnet` | `ghcr.io/aurelius-protocol/aurelius-miner:latest` |
| Simulation sidecar | `…/aurelius-concordia:testnet` (pulled automatically) | `…/aurelius-concordia:latest` |

The published image is the supported path for running a validator. The validator's
stage-7 pipeline launches sandboxed Concordia simulation containers via a Docker socket,
so it expects to run inside a container that has access to the host's Docker daemon via
the socket proxy — which the quickstart below sets up for you. Source checkouts are great
for development, CI, and reading the code; for operating a validator, the image is the
path that has working simulation out of the box.

Every push to `main` rebuilds the mainnet `:latest` tag; every push to `testnet` rebuilds
the `:testnet` tag.

---

## Quickstart — Mainnet validator (SN 37)

Prerequisites:

- Docker 20.10+ and `docker compose`
- A Bittensor wallet registered on mainnet `netuid 37`
  (`btcli subnet register --netuid 37 --network finney`) — registration costs TAO
- An OpenAI-compatible LLM API key — [DeepSeek](https://platform.deepseek.com/) is the
  default and cheapest; OpenAI, Anthropic, and other compatible providers also work

### 1. Write a minimal `.env`

These four variables are the full operator-side config. The `ENVIRONMENT` profile
auto-selects subnet, network, Central API URL, simulation resources, and safety flags —
setting any of those directly is almost never necessary.

```bash
cat > .env <<'EOF'
ENVIRONMENT=mainnet
WALLET_NAME=<your-wallet>
WALLET_HOTKEY=<your-hotkey>
LLM_API_KEY=<your-openai-compatible-api-key>
EOF
```

If you're adapting an older `.env` from a previous release, it's easier to start from this
minimal template than to prune the old one — a legacy variable like `CENTRAL_API_URL=` or
`BT_NETUID=` can override the profile default if it's still present. See
[Configuration](#configuration) for the full list and when an override makes sense.

### 2. Docker compose with socket-proxy sidecar

The validator reaches the Docker daemon through
[`tecnativa/docker-socket-proxy`](https://github.com/Tecnativa/docker-socket-proxy), which
restricts the socket to only the API calls the validator actually uses.

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
Central API`, `Clock drift check passed`, `Remote config refreshed`, and a `Config
summary` line ending with `env=mainnet network=finney
api_url=https://new-collector-api-production.up.railway.app llm_model=deepseek-chat …
burn_mode=True`. Cycle summaries then print every few minutes.

If any of those are missing or followed by warnings, see [Troubleshooting](#troubleshooting).

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

Register on `netuid 455` on `network test`, and swap `:latest` → `:testnet` in the
compose file. Everything else — socket proxy, volumes, labels — is identical.

---

## Quickstart — Miner

Miners don't run simulations and don't need the Docker-socket proxy, so a single command
is enough.

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

The axon must be reachable on the public IP and port you advertise — validators can only
query miners they can connect to.

---

## Auto-update via Watchtower (optional)

Add this to the compose file alongside `aurelius-validator` to auto-pull new images every
5 minutes:

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

The validator isn't reaching the Central API. The `Config summary` line a few lines below
the warning shows the `api_url=` it's pointing at, which usually points to the cause:

| `api_url` shows | What that means | Fix |
|---|---|---|
| `http://localhost:8000` | `.env` has `ENVIRONMENT=local` (local profile's default) | Set `ENVIRONMENT=mainnet` (or `testnet`) |
| `…-staging.up.railway.app` | `.env` has `ENVIRONMENT=testnet` but you want mainnet | Set `ENVIRONMENT=mainnet` |
| A URL you don't recognize, or empty | `.env` has an explicit `CENTRAL_API_URL=…` entry overriding the profile default | Remove that line; the profile will fill it in |
| The expected production URL | Network path to Railway is blocked | Verify with `curl -v https://new-collector-api-production.up.railway.app/health` from inside the container |

A note on empty values: `CENTRAL_API_URL=` with nothing after the `=` is treated as an
explicit empty-string override by Python's env-reading conventions. Recent validator
images fall back to the profile default when the value is empty, but older images do not.
The safest pattern is to omit the line entirely rather than leave it blank.

### `Config summary | env=local network=test …` on a mainnet host

`.env` has `ENVIRONMENT=local`. The local profile uses `http://localhost:8000`, the
testnet network, and enables testlab mode — none of which are right for SN 37. Setting
`ENVIRONMENT=mainnet` fixes all four (api_url, network, netuid, testlab flag) at once.

### `Failed to set weights: Hotkey … not registered in subnet 37`

The hotkey needs to be registered on the subnet. Register with:

```bash
btcli subnet register --netuid 37 --network finney \
  --wallet.name <name> --wallet.hotkey <hotkey>
```

(Registration costs TAO.)

### `Failed to persist ramp-up anchor … Permission denied: '/app/data/…'`

The `./data` bind-mount on the host is owned by a different uid than the container's
`appuser` (uid 1000). Align them:

```bash
chown -R 1000:1000 ./data ./simdata
```

### Dependency notes for source builds

If you're running from a source checkout and see one of these at import time, the lock
file gives you the tested dep combination:

- `RuntimeError: Conflict detected: 'scalecodec' … conflicts with 'cyscale'` —
  `async-substrate-interface` 2.x added a conflict check that trips when scalecodec is
  also present. The lock pins `async-substrate-interface==1.6.3`, which doesn't include
  the check.
- `ImportError: cannot import name 'ScaleObj' from 'async_substrate_interface.types'` —
  bittensor newer than `10.2.x` imports `ScaleObj`, which isn't in the 1.6.x line. The
  lock pins `bittensor==10.2.0`, which doesn't need it.

Either way: `pip install -r requirements.lock` before `pip install -e .` to anchor the
resolution to known-good versions.

---

## How it works

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

The `ENVIRONMENT` profile (`local` / `testnet` / `mainnet`) sets subnet, network, Central
API URL, simulation resources, and safety flags. Operators normally only set the four
variables in the quickstart; the rest come from the profile.

Required for both validators and miners:

| Variable | Purpose | Default |
|---|---|---|
| `ENVIRONMENT` | `local` \| `testnet` \| `mainnet` | `local` |
| `WALLET_NAME` | Bittensor coldkey wallet name | `default` |
| `WALLET_HOTKEY` | Bittensor hotkey name | `default` |

Validator-only:

| Variable | Purpose | Default |
|---|---|---|
| `LLM_API_KEY` | OpenAI-compatible LLM key for Concordia | (empty — required to run simulations) |

Miner-only:

| Variable | Purpose | Default |
|---|---|---|
| `AXON_EXTERNAL_IP` | Public IP the miner advertises | (empty → use local IP) |
| `AXON_EXTERNAL_PORT` | Public port the miner advertises | `8091` |

Optional overrides — set only to replace a profile default:

| Variable | Default | Set when |
|---|---|---|
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | Using a non-DeepSeek LLM provider |
| `LLM_MODEL` | `deepseek-chat` | Using a non-default model |

See [`aurelius/config.py`](aurelius/config.py) for authoritative per-profile defaults and
the remaining knobs (simulation tuning, timeouts, queue sizes, etc.).

**Two-tier config model.** The variables above are the *local* tier — wallet, network,
secrets, set once at startup. A *remote* tier (polling interval, classifier threshold,
novelty threshold, rate limits, minimum protocol versions) is fetched from the Central
API at runtime, cached for 5 minutes, and refreshed transparently. Operators don't set
remote-tier values.

---

## Development

Development and CI typically run from a source checkout. Pairing the lock file with the
editable install keeps dep resolution aligned with what the published image builds from:

```bash
git clone https://github.com/Aurelius-Protocol/Aurelius-Protocol.git
cd Aurelius-Protocol

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.lock
pip install -e ".[ml,simulation,dev]"

cp .env.example .env
$EDITOR .env                     # ENVIRONMENT=local for a testlab loop

aurelius-validator               # or: aurelius-miner
```

Operating a mainnet or testnet validator is easiest with the published image — the
simulation stage expects the image's Docker-socket plumbing, which isn't set up
automatically in a source environment.

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

## Security notes

- **Concordia isolation.** Every simulation runs in an ephemeral Docker container with
  capped RAM / CPU, egress limited to `SIM_ALLOWED_LLM_HOSTS`, and no persistent
  filesystem outside the mounted `/sim-data`.
- **Socket proxy.** The quickstart uses `tecnativa/docker-socket-proxy` so the validator
  container can only invoke the Docker API calls it actually needs (`CONTAINERS`,
  `IMAGES`, `NETWORKS`, `POST`). Mounting `/var/run/docker.sock` directly would give the
  validator container full host-daemon access — the proxy is the recommended boundary
  for long-running deployments.
- **Image digest pinning.** `REQUIRE_IMAGE_DIGEST=1` is on by default in the testnet and
  mainnet profiles. CI auto-pins the Concordia image digest after each build, so
  operators don't need to configure `CONCORDIA_IMAGE_DIGEST` themselves — keeping the
  validator image current is enough.
- **Work-token accounting.** Balance is checked in stage 3 and deducted only in stage 8
  after a successful simulation. If the Central API is unreachable during the balance
  check, submissions are rejected rather than admitted for free.

---

## Links

- [Bittensor docs](https://docs.bittensor.com)
- [Subnet 455 on taostats (testnet)](https://taostats.io/subnet/455/)
- [Subnet 37 on taostats (mainnet)](https://taostats.io/subnet/37/)
- [GHCR packages](https://github.com/orgs/Aurelius-Protocol/packages)
- [Issues](https://github.com/Aurelius-Protocol/Aurelius-Protocol/issues)

## License

MIT
