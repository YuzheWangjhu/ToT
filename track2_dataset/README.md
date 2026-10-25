# Track 3: Long-horizon conversational navigation

The evaluated dataset is `generated_data/track2_role_ipc_test_v2/dataset`: 22 recordings and 418 tasks. Frozen metadata is included; acquire source audio separately and reconstruct it using the complete workflow in [datasets.md](../docs/datasets.md).

From the repository root:

```bash
python scripts/verify_datasets.py
python scripts/reconstruct_track3.py --dataset-root /path/to/seamless/datasets
python scripts/verify_datasets.py --audio --group seamless
```

The reconstruction wrapper validates source checksums, selected scenes, v1 audio, accepted human corrections, task answers and the exact evaluated fingerprint. The shared benchmark CLI runs this track with `--tracks 2`. See [the root README](../README.md) for model configuration and conditions.
