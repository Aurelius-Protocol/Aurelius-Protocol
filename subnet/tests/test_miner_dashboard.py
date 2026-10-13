import json
import threading
import urllib.error
import urllib.request

import pytest

from subnet.miner.dashboard import server
from subnet.miner.journal import AttemptJournal

pytestmark = pytest.mark.miner


def test_loopback_read_only_and_rebinding_protection(tmp_path):
    journal = AttemptJournal(tmp_path / "journal.sqlite3")
    http = server(journal, port=0, pool=tmp_path / "pool")
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{http.server_port}"
    try:
        with urllib.request.urlopen(url + "/api/attempts") as response:
            assert json.load(response) == {"attempts": [], "pool": [], "experiments": []}
            assert response.headers["Cache-Control"] == "no-store"
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        for headers in ({"Host": "evil.example"}, {"Origin": "https://evil.example"}):
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(urllib.request.Request(url, headers=headers))
            assert caught.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(urllib.request.Request(url + "/api/attempts", data=b"{}"))
        assert caught.value.code == 501
    finally:
        http.shutdown()
        http.server_close()
        thread.join()
