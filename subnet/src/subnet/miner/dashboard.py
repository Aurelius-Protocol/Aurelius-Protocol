"""Read-only loopback dashboard. No wallet loading, remote requests, or submission controls."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sqlite3
from pathlib import Path
from .staging import StagingPool, item_id

from .journal import AttemptJournal

HTML = b"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Miner observations</title>
<link rel="stylesheet" href="/style.css"><main><h1>Miner observations</h1>
<p>Local journal. Run <code>aurelius-miner observe</code> to refresh remote charge/refund and validator observations. Missing evidence is unknown.</p>
<p id="state">Loading journal...</p><section id="attempts"></section><h2>Staging pool</h2><pre id="pool"></pre><h2>Local experiments</h2><section id="experiments"></section></main><script src="/app.js"></script></html>"""
CSS = b"""body{font:16px system-ui;margin:2rem;background:#101722;color:#edf3fa}main{max-width:1100px;margin:auto}
article{background:#1a2635;padding:1.3rem;margin:1rem 0;border-radius:12px}h2{font-size:1.1rem}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}summary{cursor:pointer;padding:.5rem}
small{color:#b6c6d8} .facts{display:flex;gap:2rem;flex-wrap:wrap}"""
JS = b""""use strict";
function text(tag,value,parent){const e=document.createElement(tag);e.textContent=value;parent.appendChild(e);return e;}
async function update(){try{const r=await fetch('/api/attempts',{cache:'no-store'});if(!r.ok)throw Error('Journal unavailable');
const data=await r.json();document.getElementById('pool').textContent=JSON.stringify(data.pool,null,2);
const experiments=document.getElementById('experiments');experiments.replaceChildren();
for(const run of data.experiments){const d=document.createElement('details');experiments.appendChild(d);
text('summary',run.name+' / '+run.report.status,d);text('pre',JSON.stringify(run.report,null,2),d);}
const root=document.getElementById('attempts');root.replaceChildren();
document.getElementById('state').textContent='Read at '+new Date().toLocaleString()+'; '+data.attempts.length+' attempts';
for(const a of data.attempts){const box=document.createElement('article');root.appendChild(box);
text('h2',(a.identity.slot_id||a.identity.task_type)+' / epoch '+a.identity.slot_epoch+' / window '+a.identity.fanout_window,box);
text('small',a.identity.content_hash,box);const facts=document.createElement('div');facts.className='facts';box.appendChild(facts);
const last=a.events.at(-1);text('p','Last observation: '+(last?last.stage:'prepared'),facts);
const charge=a.events.filter(e=>e.stage==='charge_observation').at(-1);
text('p','Charge: '+(charge?charge.data.charge:'unknown'),facts);text('p','Refund: '+(charge?charge.data.refund:'unknown'),facts);
for(const e of a.events){const d=document.createElement('details');box.appendChild(d);
text('summary',e.stage+' / '+e.source+' / '+new Date(e.observed_at*1000).toLocaleString(),d);text('pre',JSON.stringify(e.data,null,2),d);}
const d=document.createElement('details');box.appendChild(d);text('summary','Exact identity and upload acknowledgement',d);
text('pre',JSON.stringify({identity:a.identity,upload:a.upload},null,2),d);}}
catch(e){document.getElementById('state').textContent='Journal unavailable; displayed observations may be stale';}}
update();setInterval(update,10000);"""


def experiment_reports(directory):
    if directory is None:
        return []
    root = Path(directory)
    reports = []
    for path in sorted(root.glob("*/report.json"))[:64]:
        if path.is_symlink() or path.parent.is_symlink():
            continue
        with path.open("rb") as handle:
            raw = handle.read(2 * 1024**2 + 1)
        if len(raw) > 2 * 1024**2:
            continue
        report = json.loads(raw)
        if isinstance(report, dict) and report.get("kind") in {
            "sae_local_training",
            "sae_local_evaluation",
            "sae_local_comparison",
        }:
            reports.append({"name": path.parent.name, "report": report})
    return reports


def server(journal: AttemptJournal, *, port: int = 8093, pool=None, experiments=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected or self.headers.get("Origin") not in {
                None,
                f"http://{expected}",
            }:
                self.send_error(403)
                return
            if self.path == "/api/attempts":
                try:
                    staging = StagingPool(pool)
                    enabled = staging.enabled()
                    data = json.dumps(
                        {
                            "attempts": journal.attempts(),
                            "pool": [
                                {
                                    "item_id": item_id(item),
                                    "enabled": enabled.get(item_id(item), True),
                                    "candidate": item,
                                }
                                for item in staging.items()
                            ],
                            "experiments": experiment_reports(experiments),
                        },
                        allow_nan=False,
                    ).encode()
                except (OSError, ValueError, RuntimeError, sqlite3.Error):
                    self.send_error(503)
                    return
                mime = "application/json"
            elif self.path in {"/", "/app.js", "/style.css"}:
                data, mime = {
                    "/": (HTML, "text/html"),
                    "/app.js": (JS, "text/javascript"),
                    "/style.css": (CSS, "text/css"),
                }[self.path]
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(journal=None, *, port=8093, pool=None, experiments=None):
    with server(AttemptJournal(journal), port=port, pool=pool, experiments=experiments) as http:
        print(f"Miner observations: http://127.0.0.1:{http.server_port}", flush=True)
        http.serve_forever()
