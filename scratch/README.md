# scratch/

One-off and AI-generated code: probes, inspection snippets, report generators,
smoke-test harnesses. This is the **default home for new experimental code** —
things live here until they earn promotion into `src/` (reusable, reviewed
logic) or `scripts/` (pipeline drivers we expect to rerun).

Rules:

- **Nothing outside `scratch/` may import from it.**
- Anything here may be stale, broken, or superseded — check dates before reuse.

Contents:

- `vertex_smoke_test/` — minimal Docker image + loop script used to verify the
  Vertex AI job wrapper (`scripts/vertex_job_runner.py`) end to end before
  running real experiments.
- `notes/` — private analysis writeups and research notes (gitignored;
  formerly the top-level `local/` directory).
