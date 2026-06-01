# Dataset Construction Pipeline

Canonical pipeline for creating tool relevance datasets.

## Stages

1. Build or merge tool universes.
2. Generate synthetic plugin metadata when needed.
3. Generate conversations.
4. Sample candidate tool sets.
5. Judge candidate sets.
6. Calibrate scores.
7. Produce quality reports.

## Files

- `source/`: pipeline implementation copied into the clean workspace.
- `configs/`: example or environment-driven configs only.
- `docs/`: design documents.
- `scripts/`: helper scripts.

## Security Rule

Do not store real API keys in config files. Use environment variables.
