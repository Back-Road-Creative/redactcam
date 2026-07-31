"""Model download-on-demand cache — no real network, no real weights.

The payload is a handful of bytes standing in for an ONNX file; every URL is
fake and ``urlopen`` is always stubbed, so this suite never reaches the network.
"""

import hashlib
import io

import pytest

from redactcam import models
from redactcam.models import ModelSpec

PAYLOAD = b"not-really-onnx-bytes" * 100
SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "models"


def _fake_urlopen(payload):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    def _open(req, timeout=None):
        return _Resp(payload)

    return _open


class TestOverride:
    def test_existing_path_wins(self, tmp_path, cache):
        m = tmp_path / "my.onnx"
        m.write_bytes(b"x")
        spec = ModelSpec(name="plate", url="https://h/m.onnx", sha256=SHA, path=str(m))
        assert models.resolve_model(spec, cache) == m

    def test_missing_path_returns_none_without_downloading(self, tmp_path, cache, caplog):
        spec = ModelSpec(name="plate", path=str(tmp_path / "nope.onnx"))
        with caplog.at_level("WARNING"):
            assert models.resolve_model(spec, cache) is None
        assert "does not exist" in caplog.text


class TestConfiguration:
    def test_unconfigured_spec_returns_none(self, cache):
        assert models.resolve_model(ModelSpec(name="plate"), cache) is None

    def test_url_without_checksum_returns_none(self, cache):
        """A URL with no checksum is refused: fetching an unverified binary and
        running it as a model is worse than having no model."""
        spec = ModelSpec(name="plate", url="https://h/m.onnx")
        assert models.resolve_model(spec, cache) is None

    def test_non_https_refused(self, cache, caplog):
        spec = ModelSpec(name="plate", url="http://h/m.onnx", sha256="ab")
        with caplog.at_level("WARNING"):
            assert models.resolve_model(spec, cache) is None
        assert "not https" in caplog.text


class TestDownload:
    SPEC = ModelSpec(name="plate", url="https://h/m.onnx", sha256=SHA)

    def test_downloads_verifies_caches(self, cache, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(PAYLOAD))
        path = models.resolve_model(self.SPEC, cache)
        assert path is not None and path.exists()
        assert models._sha256(path) == SHA

    def test_cache_hit_skips_download(self, cache, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(PAYLOAD))
        first = models.resolve_model(self.SPEC, cache)

        def _boom(*a, **k):
            raise AssertionError("a cache hit must not re-download")

        monkeypatch.setattr("urllib.request.urlopen", _boom)
        assert models.resolve_model(self.SPEC, cache) == first

    def test_checksum_mismatch_returns_none(self, cache, monkeypatch, caplog):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(PAYLOAD))
        spec = ModelSpec(name="plate", url="https://h/m.onnx", sha256="00" * 32)
        with caplog.at_level("WARNING"):
            assert models.resolve_model(spec, cache) is None
        assert "checksum mismatch" in caplog.text

    def test_mismatched_download_leaves_nothing_behind(self, cache, monkeypatch):
        """A rejected download must not litter the cache with a partial file the
        next run could pick up."""
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(PAYLOAD))
        spec = ModelSpec(name="plate", url="https://h/m.onnx", sha256="00" * 32)
        models.resolve_model(spec, cache)
        assert list(cache.glob("*")) == []

    def test_network_error_degrades(self, cache, monkeypatch, caplog):
        def _boom(*a, **k):
            raise OSError("network down")

        monkeypatch.setattr("urllib.request.urlopen", _boom)
        with caplog.at_level("WARNING"):
            assert models.resolve_model(self.SPEC, cache) is None
        assert "fetch failed" in caplog.text


class TestDefaults:
    def test_every_default_is_https_with_a_checksum(self):
        """No default may fetch over plain HTTP or without a checksum."""
        for kind, spec in models.DEFAULT_MODELS.items():
            assert spec.name == kind
            assert spec.url.startswith("https://"), kind
            assert len(spec.sha256) == 64, kind
            assert spec.path == "", kind  # no machine-specific path baked in

    def test_no_weights_are_vendored(self):
        """The package ships code, not model binaries."""
        pkg = __import__("redactcam").__path__[0]
        from pathlib import Path

        assert list(Path(pkg).rglob("*.onnx")) == []

    def test_cache_name_is_checksum_scoped(self, cache, monkeypatch):
        """Two models with different checksums never collide in the cache, so a
        model update cannot be masked by a stale file of the same name."""
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(PAYLOAD))
        path = models.resolve_model(ModelSpec("face", "https://h/f.onnx", SHA), cache)
        assert path.name == f"face_{SHA[:12]}.onnx"
