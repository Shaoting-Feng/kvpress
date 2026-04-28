# AA-LCR × kvpress × gpt-oss-120b

Quality-vs-TTFT trade-off study using the
[AA-LCR](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR) long-context
benchmark (100 questions over ~100k-token document sets), `openai/gpt-oss-120b`,
and kvpress KV-cache compression on this `gptoss` branch.

**Read [HANDOFF.md](HANDOFF.md) first** — it has the experiment design,
decisions log, file inventory, smoke recipe, full-sweep command, and a
catalogue of known gotchas.

This directory is **self-contained**: profiling/testing splits, the AA-LCR
document corpus, the StorageManager (`components.py`), the F1 scorer, the
runner, the simulation, and the speed-profile approximation are all bundled.
