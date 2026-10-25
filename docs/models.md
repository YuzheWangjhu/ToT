# Model configurations and adapters

Use the ten configurations below to match the evaluated model set. Each configuration records the model identifier and decoding settings used by the benchmark. A provider alias or a model repository name is not an immutable model revision; exact upstream revisions and complete model-specific runtime locks were not recorded in these configuration files. Archived outputs are the basis for exact paper-result reproduction.

| Paper label | Configuration | Model identifier |
| --- | --- | --- |
| GPT | `configs/gpt_audio/config.json` | `gpt-audio-2025-08-28` |
| GPT-m | `configs/gpt_audio_mini/config.json` | `gpt-audio-mini` |
| Gem-2.5 | `configs/gemini_2_flash/config.json` | `gemini-2.5-flash` |
| Gem-3 | `configs/gemini_3_flash/config.json` | `gemini-3-flash-preview` |
| AFN | `configs/audio_flamingo_next/config.json` | `nvidia/audio-flamingo-next-hf` |
| AFN-T | `configs/audio_flamingo_next_think/config.json` | `nvidia/audio-flamingo-next-think-hf` |
| Kimi | `configs/kimi_audio/config.json` | `moonshotai/Kimi-Audio-7B-Instruct` |
| MiMo | `configs/mimo_audio/config.json` | `XiaomiMiMo/MiMo-Audio-7B-Instruct` |
| MOSS | `configs/moss_audio/config.json` | `OpenMOSS-Team/MOSS-Audio-8B-Instruct` |
| Qwen | `configs/qwen25_omni/config.json` | `Qwen/Qwen2.5-Omni-7B` |

These ten configurations are the evaluated model set included in this repository. The dummy adapter is used only for offline implementation checks.

## Run a new model evaluation

Fresh evaluations are optional and are not needed to reproduce the archived results. Follow [dataset verification and reconstruction](datasets.md), then install the requirements for the selected adapter below.

For an API model, prepare the local credential file from the repository root:

```bash
python -m pip install -r requirements/api.txt
mkdir -p .env
cp -n .env.example .env/api_keys.env
```

Fill in the selected provider's credentials in `.env/api_keys.env`, then run, for example:

```bash
python run_theory_of_time_benchmark.py run \
  --dataset-root generated_data/theory_of_time_v1 \
  --results-root runs/gpt_audio \
  --adapter-config configs/gpt_audio/config.json \
  --tracks 1a 1b 1c \
  --conditions active full_context \
  --time-mode binned
```

This command makes paid API calls. For the conversational same-session condition, use `--tracks 1c --conditions active_same_session_periodic_probe`. After reconstructing the long-horizon audio, run:

```bash
python run_theory_of_time_benchmark.py run \
  --track2-dataset-root track2_dataset/generated_data/track2_role_ipc_test_v2 \
  --results-root runs/gpt_audio \
  --adapter-config configs/gpt_audio/config.json \
  --tracks 2 \
  --conditions active
```

Use a new results directory for each configuration. Provider updates and unpinned model revisions can cause new runs to differ from the archived outputs. `python run_theory_of_time_benchmark.py run --help` lists the available filters.

## Runtime and credentials

Install core dependencies with `python -m pip install -r requirements.txt`, then install only the runtime requirements for the selected adapter from `requirements/`. API adapters use `requirements/api.txt`. Local adapters require their model-specific libraries and model weights; the core or analysis environment alone does not install every model runtime. The paper's local runs used a single A100 GPU. Memory requirements depend on the model and audio length.

The blank `.env.example` documents expected credential names. Create `.env/api_keys.env` locally and populate `OPENAI_API_KEY` for GPT or `GOOGLE_API_KEY`/`GEMINI_API_KEY` for Gemini. Relative `api_key_env_path` values resolve from the repository root, regardless of where the config is stored or which working directory invokes the runner. Absolute credential paths are used unchanged. Keep credentials out of configs, logs, notebooks, and commits. Model weights and external model code retain their own terms and are acquired separately.

API settings use temperature 0 and top-p 1.0 where supported. Local settings use greedy decoding with sampling disabled. Standard output limits are 256 tokens; AFN-T uses 1,024, a repetition penalty of 1.2, and removes emitted reasoning content before scoring. Gemini 3 Flash uses the high thinking level. MiMo disables thinking and specifies the separate `XiaomiMiMo/MiMo-Audio-Tokenizer` tokenizer. Paper Track 3 overrides the action/final limits to 512/1,024 tokens.

## Long-horizon capabilities

Configuration keys prefixed `track2_` apply to **paper Track 3**. They declare supported conditions, strict text-only support, output-token-limit overrides, maximum audio duration, and maximum audio bytes. These declarations are validated before evaluation; they are not proof that every available provider version can process the declared maximum.

| Configurations | Declared maximum audio seconds | Declared conditions |
| --- | --- | --- |
| AFN, AFN-T | 1,800 | Active, full context, same-session |
| GPT, GPT-m, Gem-2.5, Gem-3, MiMo, MOSS | 1,200 | Active, full context, same-session |
| Kimi | 384 | Active |
| Qwen | 384 | Active, same-session |

Despite broader configuration declarations, the paper includes **Track 3 full-context results only for AFN and AFN-T**, each over all 418 task instances. Match archived condition coverage when reproducing paper tables. Treat additional newly run conditions as new experiments.

## Inspecting adapter behavior

The adapter interfaces and implementations are in `tot_benchmark/adapters.py` and `tot_benchmark/audio_model_adapters.py`. The offline tests exercise session history, audio ordering, response parsing, and capability checks.

For Track 3, inspect the five capability fields in each configuration together with `send_track2`. The adapter must honor per-call output-token limits and text-only requests. Unsupported capabilities and provider errors are recorded separately from task failures. Additional adapters or conditions constitute new experiments and are not part of the archived ten-model comparison.

## Recorded runtime specifications

Core and analysis specifications target Python 3.10. The observed MiMo and MOSS environments use Python 3.12 and must be installed separately. `provenance/environment_snapshot.json` records observed dependency versions. It is not a complete lock of the original evaluation environment, and it does not establish that each GPU setup has been validated on a new machine. The model requirement files pin the observed Kimi source commit and Audio Flamingo Transformers fork. Weight revisions and some external source revisions remain unrecorded.

| Adapter | Dependency specification | Additional source requirement |
| --- | --- | --- |
| Qwen | `requirements/models/qwen25_omni.txt` | Transformers Qwen Omni implementation |
| AFN and AFN-T | `requirements/models/audio_flamingo_next.txt` | Pinned Audio Flamingo Transformers fork, included in the specification |
| Kimi | `requirements/models/kimi_audio.txt` | Pinned Kimi source package, included in the specification |
| MiMo | `requirements/models/mimo_audio.txt` | Upstream MiMo-Audio source providing `src.mimo_audio.mimo_audio.MimoAudio` |
| MOSS | `requirements/models/moss_audio.txt` | Upstream MOSS-Audio source providing `src.modeling_moss_audio` and `src.processing_moss_audio` |

For MiMo and MOSS, acquire the upstream source under its own terms and install it or add its root to `PYTHONPATH`. The wrapper source is not embedded in this repository or installed by the package-only requirements. Keep these models in separate environments with one upstream `src` tree visible at a time. Before evaluation, verify the expected import in that environment. The adapter also supports the upstream top-level module forms shown in `tot_benchmark/audio_model_adapters.py`. Record the acquired source commit and weight revision alongside any new run.

## Maintaining evaluation runs

`python scripts/rerun_api_errors.py --help` describes how to rerun unresolved soft API errors. Supply the configuration explicitly with `--adapter-config configs/<model>/config.json`; it is independent of `--model-dir`. The model directory must be inside the repository and contain `results/` for binned time or `results_continuous/` for continuous time. For example, a run created with `--results-root runs/my_experiment/results` uses `--model-dir runs/my_experiment`. `--dry-run` lists selected instances without making model calls. Rerunning without it can make paid API calls and update that run's outputs.

`python scripts/archive_api_error_results.py --help` describes the companion archival command. It uses the same result-directory layout, supports `--dry-run`, and checks SLURM for active API jobs. Use these tools on your own runs; the frozen archive remains the input for paper reproduction.
