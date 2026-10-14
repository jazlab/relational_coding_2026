# Behavior Processing

This directory turns the DANDI behavior/task NWB files into one table,
`cache/behavior_processing/ring.csv`, with one row per trial. Every figure reads
that table.

To build it, download the behavior data (`python download_dandi_data.py
--modality=behavior` from the repo root), then run:
```
python behavior_processing/run_cache_data.py
```
This takes a few seconds. See the docstring at the top of `run_cache_data.py`
for the column conventions.
