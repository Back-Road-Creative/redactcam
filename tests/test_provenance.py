"""Processing identity: what exactly ran, and is the installed copy the one the
caller thinks it is.

A render receipt that says "redactcam 0.2.0" is not an identity -- the version
string did not change across the tracking fix (#10), so two very different
builds carried the same number. The identity is the hash of the code itself,
the commit it was installed from when that is knowable, and the hash of every
model file the run resolved.
"""

import json

import pytest

from redactcam import provenance as pv


class TestCodeHash:
    def test_is_a_sha256_over_the_package_sources(self):
        digest, count = pv.code_sha256()
        assert digest is not None and len(digest) == 64
        assert count >= 10

    def test_changes_when_a_source_file_changes(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        before, _ = pv.code_sha256(tmp_path)
        (tmp_path / "a.py").write_text("x = 2\n")
        after, _ = pv.code_sha256(tmp_path)
        assert before != after

    def test_line_endings_do_not_change_it(self, tmp_path):
        """A Windows checkout with autocrlf must hash the same as the wheel."""
        (tmp_path / "a.py").write_bytes(b"x = 1\ny = 2\n")
        lf, _ = pv.code_sha256(tmp_path)
        (tmp_path / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
        crlf, _ = pv.code_sha256(tmp_path)
        assert lf == crlf

    def test_bytecode_caches_are_ignored(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        before, _ = pv.code_sha256(tmp_path)
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "a.py").write_text("junk")
        assert pv.code_sha256(tmp_path)[0] == before

    def test_no_sources_on_disk_is_unknown_not_a_hash_of_nothing(self, tmp_path):
        """A frozen build has no .py files. An empty digest would look like an
        identity and match every other frozen build."""
        assert pv.code_sha256(tmp_path) == (None, 0)


class TestInstalledRevision:
    def test_reads_the_commit_a_vcs_install_recorded(self, monkeypatch):
        class Dist:
            def read_text(self, name):
                assert name == "direct_url.json"
                return json.dumps(
                    {
                        "url": "https://user:secret@example.invalid/redactcam",
                        "vcs_info": {"vcs": "git", "commit_id": "6d731a5fb" + "0" * 31},
                    }
                )

        monkeypatch.setattr(pv.importlib.metadata, "distribution", lambda name: Dist())
        rev = pv.installed_revision()
        assert rev["commit"].startswith("6d731a5fb")
        assert rev["source"] == "direct_url"
        # the URL can carry credentials; it must never reach a receipt
        assert "secret" not in json.dumps(rev)

    def test_unknown_when_nothing_records_it(self, monkeypatch, tmp_path):
        class Dist:
            def read_text(self, name):
                return None

        monkeypatch.setattr(pv.importlib.metadata, "distribution", lambda name: Dist())
        rev = pv.installed_revision(tmp_path)  # tmp_path has no .git either
        assert rev == {"commit": None, "source": None}


class TestModelIdentity:
    def test_hashes_each_resolved_file(self, tmp_path):
        a = tmp_path / "face.onnx"
        a.write_bytes(b"face-bytes")
        out = pv.model_identity({"face": a, "plate": None})
        assert out["face"]["sha256"] == pv.sha256_file(a)
        assert out["face"]["size_bytes"] == len(b"face-bytes")
        assert out["plate"] is None  # unresolved is recorded as such, not dropped

    def test_a_shared_file_is_reported_for_both_kinds(self, tmp_path):
        coco = tmp_path / "coco.onnx"
        coco.write_bytes(b"coco")
        out = pv.model_identity({"vehicle": coco, "person": coco})
        assert out["vehicle"]["sha256"] == out["person"]["sha256"]


    def test_an_unreadable_file_is_an_unknown_hash_not_a_crash(self, tmp_path):
        out = pv.model_identity({"face": tmp_path / "gone.onnx"})
        assert out["face"]["sha256"] is None


class TestRuntimeIdentity:
    def test_is_bounded_json_with_the_named_parts(self, monkeypatch):
        monkeypatch.setattr(pv, "tool_status", lambda n: {"name": n, "version": f"{n} 1.0"})
        ident = pv.runtime_identity()
        json.dumps(ident)  # serialisable
        assert ident["redactcam"]["version"]
        assert len(ident["redactcam"]["code_sha256"]) == 64
        assert ident["native"]["opencv"]
        assert ident["native"]["ffmpeg"] == "ffmpeg 1.0"
        assert len(json.dumps(ident)) < 4000


class TestCheckCurrent:
    @pytest.fixture(autouse=True)
    def _fixed_identity(self, monkeypatch):
        monkeypatch.setattr(
            pv,
            "runtime_identity",
            lambda: {
                "redactcam": {
                    "version": "0.2.0",
                    "code_sha256": "ab" * 32,
                    "revision": {"commit": "c34854e" + "0" * 33, "source": "direct_url"},
                }
            },
        )

    def test_a_match_returns_the_identity(self):
        ident = pv.check_current(expected_revision="c34854e", expected_version="0.2.0")
        assert ident["redactcam"]["version"] == "0.2.0"

    def test_a_different_commit_is_stale(self):
        with pytest.raises(pv.StaleInstallError, match="6d731a5"):
            pv.check_current(expected_revision="6d731a5fb")

    def test_a_different_code_hash_is_stale_even_at_the_same_version(self):
        with pytest.raises(pv.StaleInstallError, match="code_sha256"):
            pv.check_current(expected_code_sha256="cd" * 32)

    def test_an_unknowable_revision_is_not_a_pass(self, monkeypatch):
        monkeypatch.setattr(
            pv,
            "runtime_identity",
            lambda: {
                "redactcam": {
                    "version": "0.2.0",
                    "code_sha256": None,
                    "revision": {"commit": None, "source": None},
                }
            },
        )
        with pytest.raises(pv.StaleInstallError, match="cannot be determined"):
            pv.check_current(expected_revision="c34854e")

    def test_a_short_prefix_is_refused(self):
        """Three hex characters match too many commits to mean anything."""
        with pytest.raises(ValueError, match="at least 7"):
            pv.check_current(expected_revision="c34")

    def test_needs_something_to_compare_against(self):
        with pytest.raises(ValueError):
            pv.check_current()


def test_write_receipt_is_atomic_json(tmp_path):
    path = tmp_path / "sub" / "r.json"
    pv.write_receipt(path, {"schema": "x", "n": 1})
    assert json.loads(path.read_text()) == {"schema": "x", "n": 1}
    assert not list(path.parent.glob("*.tmp"))
