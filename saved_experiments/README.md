# Saved Experiments

Use this folder for experiment snapshots you want to keep in git.

> **Note:** archived snapshots (and all other run artifacts) now live in the
> public HF dataset repo
> [kunwar45/obfuscation-prompting](https://huggingface.co/datasets/kunwar45/obfuscation-prompting)
> under `saved_experiments/` — local copies were removed 2026-08-05. Push new
> snapshots there with
> `hf upload kunwar45/obfuscation-prompting saved_experiments saved_experiments --repo-type dataset`.

Everything in the repo's normal output locations stays ignored by default:

- `results/`
- `data/`
- `activations/`
- `vertex_downloads/`
- generated `vertex_jobs/runs/*.yaml`

If an experiment is worth preserving, copy or move its files into a dedicated
subfolder here, for example:

```text
saved_experiments/
  iman-2026-04-03-last-token-rerun/
    README.md
    run_config.yaml
    results/
    data/
    activations/
```

Suggested contents for each saved experiment folder:

- `README.md`: short note on what the run was and why it matters
- `run_config.yaml`: the exact Vertex config or local config used
- `results/`: selected result artifacts
- `data/`: the dataset snapshot for that run, if needed
- `activations/`: only if you intentionally want them in git

Recommended workflow:

1. Run experiments normally using the ignored output folders.
2. Decide which runs are worth keeping.
3. Copy the exact files you want to preserve into `saved_experiments/<owner>-<name>/`.
4. Commit only that saved snapshot.

Recommended naming:

- Vertex run configs and display names: `<owner>-<run-name>`
- Saved experiment folders: `<owner>-<date>-<short-description>`

This keeps day-to-day experiments out of git while making important runs easy
to preserve reproducibly.
