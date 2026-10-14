"""Incentive-terms acknowledgement gate for `serve` and `fund`."""

import io
import json

import pytest

from subnet.miner import cli, terms

pytestmark = pytest.mark.miner


class FakeStdin(io.StringIO):
    def __init__(self, text="", tty=True):
        super().__init__(text)
        self._tty = tty

    def isatty(self):
        return self._tty


@pytest.fixture(autouse=True)
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AURELIUS_MINER_DATA", str(tmp_path))
    monkeypatch.delenv(terms.ACCEPT_ENV, raising=False)
    return tmp_path


def record(data_dir):
    return json.loads((data_dir / "incentives-acceptance.json").read_text())


def test_prompt_yes_records_acceptance_and_is_not_asked_again(data_dir):
    out = io.StringIO()
    terms.ensure_incentives_accepted(environ={}, stdin=FakeStdin("y\n"), stream=out)
    assert terms.INCENTIVES_URL in out.getvalue()
    assert record(data_dir)["version"] == terms.INCENTIVES_TERMS_VERSION
    assert record(data_dir)["method"] == "prompt"

    again = io.StringIO()
    terms.ensure_incentives_accepted(environ={}, stdin=FakeStdin(""), stream=again)
    assert again.getvalue() == ""


@pytest.mark.parametrize("answer", ["n\n", "\n", "", "maybe\n"])
def test_prompt_other_answers_refuse_and_record_nothing(data_dir, answer):
    with pytest.raises(terms.TermsNotAccepted):
        terms.ensure_incentives_accepted(environ={}, stdin=FakeStdin(answer), stream=io.StringIO())
    assert not (data_dir / "incentives-acceptance.json").exists()


def test_headless_without_opt_in_refuses_with_hint(data_dir):
    with pytest.raises(terms.TermsNotAccepted) as exc:
        terms.ensure_incentives_accepted(
            environ={}, stdin=FakeStdin("y\n", tty=False), stream=io.StringIO()
        )
    assert terms.ACCEPT_ENV in str(exc.value)
    assert terms.INCENTIVES_URL in str(exc.value)
    assert not (data_dir / "incentives-acceptance.json").exists()


def test_env_and_flag_opt_ins_record_their_method(data_dir):
    headless = FakeStdin(tty=False)
    terms.ensure_incentives_accepted(
        environ={terms.ACCEPT_ENV: "YES"}, stdin=headless, stream=io.StringIO()
    )
    assert record(data_dir)["method"] == "env"

    (data_dir / "incentives-acceptance.json").unlink()
    terms.ensure_incentives_accepted(
        accept_flag=True, environ={}, stdin=headless, stream=io.StringIO()
    )
    assert record(data_dir)["method"] == "flag"


@pytest.mark.parametrize("content", ['{"version": "2000-01-01"}', "not json", "[]"])
def test_stale_or_corrupt_record_asks_again(data_dir, content):
    (data_dir / "incentives-acceptance.json").write_text(content)
    with pytest.raises(terms.TermsNotAccepted):
        terms.ensure_incentives_accepted(
            environ={}, stdin=FakeStdin(tty=False), stream=io.StringIO()
        )


def test_fund_refuses_before_any_funding_call(monkeypatch, capsys):
    from subnet.miner import funding

    monkeypatch.setattr("sys.stdin", FakeStdin(tty=False))
    monkeypatch.setattr(funding, "run", lambda **_: pytest.fail("funding must not run"))
    code = cli.main(
        [
            "fund",
            "--alpha",
            "1",
            "--wallet-name",
            "w",
            "--wallet-path",
            "/nonexistent",
            "--intent-dir",
            "/nonexistent",
        ]
    )
    assert code == 2
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["status"] == "refused"
    assert refusal["executed"] is False
    assert terms.ACCEPT_ENV in refusal["reason"]


def test_serve_refuses_headless_without_acceptance(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", FakeStdin(tty=False))
    assert cli.main(["serve"]) == 2
    err = capsys.readouterr().err
    assert "[aurelius-miner] cannot serve: TermsNotAccepted" in err
    assert "Traceback" not in err
