# Frozen synthetic evaluation audio

**All speech in the bundled clips is AI-generated using OpenAI `gpt-4o-mini-tts`. The voices are synthetic, not recordings of human speakers.** This notice follows the [official OpenAI text-to-speech documentation](https://developers.openai.com/api/docs/guides/text-to-speech).

| Directory | Paper track | Frozen WAVs |
| --- | --- | --- |
| `track_1a/dataset/episodes/` | Track 1: Sparse acoustic temporal mapping | 10 |
| `track_1b/dataset/episodes/` | Track 1: Concurrent acoustic temporal mapping | 10 |
| `track_1c/dataset/episodes/` | Track 2: Conversational belief updating | 240 |

These are the exact evaluated clips, with their original timing and bytes. Their total size is 235,131,760 bytes (approximately 224 MiB). Paths, sizes, and SHA-256 hashes are recorded in [`provenance/dataset_audio.json`](../../provenance/dataset_audio.json). From the repository root, verify them with:

```bash
python scripts/verify_datasets.py --audio --group synthetic
```

See [dataset documentation](../../docs/datasets.md) for the annotation schema, generation recipes, and separate Seamless Interaction reconstruction workflow.
