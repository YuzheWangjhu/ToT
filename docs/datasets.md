# Frozen datasets and reconstruction

The metadata and 260 synthetic WAVs shipped here preserve the exact evaluated inputs for paper Tracks 1 and 2. All speech in these clips is **AI-generated** using OpenAI `gpt-4o-mini-tts`; the voices are not recordings of human speakers. See the [dataset notice](../generated_data/theory_of_time_v1/README.md) and [official OpenAI TTS documentation](https://developers.openai.com/api/docs/guides/text-to-speech). Seamless Interaction recordings for paper Track 3 must be reconstructed from a separately acquired corpus. Analysis of the included results does not require audio.

| Paper track | Internal ID | Frozen inventory | Location |
| --- | --- | --- | --- |
| 1 Sparse | `1a` | 10 episodes | `generated_data/theory_of_time_v1/track_1a/dataset` |
| 1 Concurrent | `1b` | 10 episodes | `generated_data/theory_of_time_v1/track_1b/dataset` |
| 2 Conversational updating | `1c` | 240 episodes: 60 conversations × four voice configurations | `generated_data/theory_of_time_v1/track_1c/dataset` |
| 3 Long-horizon navigation | `2` | 22 reels, 88 source scenes, 418 tasks | `track2_dataset/generated_data/track2_role_ipc_test_v2/dataset` |

## Integrity scopes

Run from the package root:

```bash
python scripts/verify_datasets.py
```

This checks the frozen metadata inventory, original Track 3 validation-report hashes/fingerprint, and episode/task counts. To also check all 260 bundled synthetic WAVs:

```bash
python scripts/verify_datasets.py --audio --group synthetic
```

After reconstructing the long-horizon recordings, check their audio separately:

```bash
python scripts/verify_datasets.py --audio --group seamless
```

`--audio` without `--group` requires all 282 evaluated WAVs. Missing or different files fail immediately. `provenance/dataset_audio.json` records every expected path, size and SHA-256; `provenance/dataset_files.json` records source and release metadata hashes. The recorded episode content, timing, labels, and audio paths identify the evaluated inputs. Track 3 manifests, source input records, answers and validation report remain byte-identical to the evaluated release.

## Reconstruct Track 3

Install `requirements/datasets.txt`, which provides NumPy, pandas and SoundFile. Obtain source files using the [official Seamless Interaction acquisition instructions](https://github.com/facebookresearch/seamless_interaction/blob/main/README.md) and the [official dataset distribution](https://huggingface.co/datasets/facebook/seamless-interaction), under their applicable terms. Upstream provides individual-recording and archive download methods; this release does not download the corpus automatically.

The exact acquisition contract is `provenance/dataset_source_files.json`: 176 participant WAVs plus 176 participant JSON files for 88 scenes, all from `improvised/test`. Preserve its relative layout under your chosen corpus root, for example:

```text
CORPUS_ROOT/improvised/test/0000/0001/V00_S2048_I00000691_P1306A.wav
CORPUS_ROOT/improvised/test/0000/0001/V00_S2048_I00000691_P1306A.json
```

The JSON must contain the original `metadata:transcript` and `metadata:vad` fields. File stems identify individual recordings; the two numbered directories identify the upstream batch/archive organization. Use the complete path inventory to select files or archives. Every source file is hash-checked before reconstruction, so upstream revisions cannot silently change the evaluated inputs.

```bash
python scripts/reconstruct_track3.py --dataset-root /path/to/CORPUS_ROOT
```

The command performs the complete frozen chain:

1. Validate the selected reels/scenes against the included audited candidate table.
2. Mix the two mono participant recordings at equal 0.5 gain, retain the shorter length when they differ by one sample, concatenate four scenes in frozen order with 0.5-second silence gaps, and write PCM16 WAVs using the original peak rule.
3. Compare every reconstructed v1 episode, including its audio SHA-256, with the frozen v1 manifest.
4. Apply the included human corrections, exclusions, reassigned candidates and scene-card reviews to rebuild the audited v2 tasks, answers and provenance. Check absent quotations against complete source transcripts.
5. Run the complete v2 validator and require all artifact hashes and the fingerprint to match the evaluated release before installing audio into the v2 dataset directory.

The evaluated fingerprint is:

```text
0bffa118888b587758506c0a553c0221035d7619adbf88fd73992036bdc45819
```

Reconstruction stages files in `.scratch/track3_reconstruction` by default and retains them for inspection. Existing work directories or output episode directories cause an error; use a new `--work-dir` and `--output-dir` to perform a separate reconstruction. Staging and installation use hardlinks, so those directories must be on the same filesystem. No links reference the research workspace or source corpus.

For a complete subsequent source/audit validation, supply the retained v1 WAV directory:

```bash
python track2_dataset/validate_track2_dataset_v2.py \
  --dataset-root /path/to/CORPUS_ROOT \
  --v1-audio-dir .scratch/track3_reconstruction/v1/episodes
```

`--rebuild-check` additionally compares two clean audited-v2 builds against the supplied dataset; it reuses the already reconstructed v1 audio. The end-to-end wrapper above is the check that also rebuilds that audio from the corpus. `provenance/dataset_validation.json` records the source-reconstruction checks and evaluated fingerprint.

The three accepted human-audit inputs are retained under `track2_human_audit/audit_results/`. The exact original scene metadata, candidate pools, selected reels/scenes and v1 manifest are included because v2 provenance hashes refer to them. The accepted audit records are sufficient to reconstruct the evaluated annotations.

## Synthetic frozen clips and new generation

The exact 260 evaluated synthetic clips (235,131,760 bytes, approximately 224 MiB) are included at the paths in `provenance/dataset_audio.json`; no separate download or TTS call is needed to obtain them. Their original bytes and SHA-256 hashes are preserved. Verify them with the command above. New TTS calls may produce different audio even with the same model name, voices, speed and prompt.

`scripts/generate_word_assets.py` and `scripts/word_labels.txt` preserve the controlled-track word-generation recipe; `--dry-run` lists its requests without calling the API. For an offline preview, run:

```bash
python scripts/generate_word_assets.py --output-root .scratch/new_word_assets --dry-run
```

`provenance/word_assets.json` records the source word snippets, which are not bundled and are not needed to evaluate the included episodes. The benchmark's existing dataset generation CLI composes word assets into controlled episodes. Use a new dataset root for regenerated experiments and do not replace the frozen manifests.

The conversational recipe is `track1c/build_dataset.py`; its input is `track1c/task_1c_conversation_transcripts_v10_5.csv`. It includes the four voice configurations, explicit pauses, silence trimming and normalization. To generate a new dataset with a local credential file:

```bash
python track1c/build_dataset.py \
  --csv-path track1c/task_1c_conversation_transcripts_v10_5.csv \
  --output-root generated_data/regenerated_conversations/dataset \
  --all-conversations true \
  --api-key-env-path /path/to/private/api_keys.env
```

The credential file must contain `OPENAI_API_KEY=...` and must remain untracked. Generation invokes a paid API. Generated conversational manifests contain both parsed source metadata and timing; the 240 per-episode JSON records supplied here preserve the evaluated annotations. Regeneration is a new experimental dataset and does not establish paper input reproduction.

## Schema and scoring boundary

Controlled manifests contain episode audio paths, event intervals, ground-truth rasters and behavioral answers. Conversational manifests contain conversation/version IDs, audio duration, sentence metadata and question/answer rules. Track 3 uses separate `manifest.json`, `task_instances.jsonl`, `task_answers.jsonl` and `candidate_provenance.jsonl`; scored answers and audit provenance belong to the evaluator, not model prompts. Internal family IDs remain `2A.1`, `2A.2`, `2A.3`, `2B.1`, `2B.2` with counts 110, 176, 44, 44, 44 respectively.
