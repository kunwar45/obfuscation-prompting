from pathlib import Path

from scripts import vertex_job_runner as runner


class FakeBlob:
    def __init__(self, name: str, uploads: list[tuple[str, str]]):
        self.name = name
        self.uploads = uploads

    def upload_from_filename(self, filename: str) -> None:
        self.uploads.append((self.name, filename))


class FakeBucket:
    def __init__(self, uploads: list[tuple[str, str]]):
        self.uploads = uploads

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(name, self.uploads)


class FakeClient:
    def __init__(self, uploads: list[tuple[str, str]]):
        self.uploads = uploads

    def bucket(self, _name: str) -> FakeBucket:
        return FakeBucket(self.uploads)


def test_parse_gcs_uri() -> None:
    bucket, prefix = runner.parse_gcs_uri("gs://test-bucket/obfuscation-prompting")
    assert bucket == "test-bucket"
    assert prefix == "obfuscation-prompting"


def test_sync_artifacts_uploads_metadata_and_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    uploads: list[tuple[str, str]] = []
    monkeypatch.setattr(runner.storage, "Client", lambda: FakeClient(uploads))

    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    activations_dir = tmp_path / "activations"
    results_dir.mkdir()
    data_dir.mkdir()
    activations_dir.mkdir()

    (results_dir / "run.json").write_text("{}", encoding="utf-8")
    (data_dir / "dataset.jsonl").write_text("{}", encoding="utf-8")
    (activations_dir / "acts.npz").write_text("fake", encoding="utf-8")

    metadata_path = tmp_path / ".vertex_job" / "metadata.json"
    metadata_path.parent.mkdir()
    metadata_path.write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    summary = runner.sync_artifacts(
        results_gcs_uri="gs://bucket/base",
        experiment_name="framing-experiment",
        run_slug="framing-mini-smoke/20260402_120000_deadbeef",
        artifact_dirs=["results", "data", "activations"],
        metadata_path=metadata_path,
    )

    assert summary["metadata"] == (
        "gs://bucket/base/framing-experiment/"
        "framing-mini-smoke/20260402_120000_deadbeef/metadata.json"
    )
    assert summary["artifacts"]["results"]["uploaded"] is True
    assert summary["artifacts"]["data"]["uploaded"] is True
    assert summary["artifacts"]["activations"]["uploaded"] is True

    uploaded_blob_names = {name for name, _filename in uploads}
    assert (
        "base/framing-experiment/framing-mini-smoke/"
        "20260402_120000_deadbeef/results/run.json"
    ) in uploaded_blob_names
    assert (
        "base/framing-experiment/framing-mini-smoke/"
        "20260402_120000_deadbeef/data/dataset.jsonl"
    ) in uploaded_blob_names
    assert (
        "base/framing-experiment/framing-mini-smoke/"
        "20260402_120000_deadbeef/activations/acts.npz"
    ) in uploaded_blob_names
