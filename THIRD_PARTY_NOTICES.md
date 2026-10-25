# Third-party notices

This file identifies third-party sources and applicable terms for materials used by ToT. Any license for ToT's own code does not replace third-party licenses or grant rights to external datasets, model weights, or provider services. This notice does not assign a new license to the benchmark datasets or archived model outputs.

## Seamless Interaction

Paper Track 3 (internal ID `2`) uses the **Seamless Interaction** dataset, released by Meta and its collaborators. Credit: Vasu Agrawal et al., *Seamless Interaction: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset*. The [official dataset card](https://huggingface.co/datasets/facebook/seamless-interaction) provides the complete author list and citation.

The source dataset is licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. See the [upstream license](https://huggingface.co/datasets/facebook/seamless-interaction/blob/main/LICENSE) and [license text](https://creativecommons.org/licenses/by-nc/4.0/legalcode). Reuse under that license requires attribution, a license link, identification of modifications, and compliance with its noncommercial restriction. The source material and adaptations remain subject to those terms; they are not relicensed under ToT's code license. The upstream license includes warranty disclaimers and limitations of liability. Attribution does not imply endorsement by the source creators.

Included source-derived material consists of metadata, transcript excerpts, scene descriptions, task definitions, answers, and audit records under `track2_dataset/` and `track2_human_audit/`, together with source-derived content reproduced in `artifacts/paper_results_v5.tar.gz`. The source-file inventory is [provenance/dataset_source_files.json](provenance/dataset_source_files.json).

ToT selects 88 scenes from the source dataset's `improvised/test` split, groups them into 22 reels, and constructs 418 benchmark tasks. Modifications include task construction, human annotation corrections, exclusions, and candidate reassignment. The reconstruction scripts mix participant audio at equal gain and concatenate four scenes per reel with silence gaps. Source audio and video are not bundled. See [dataset documentation](docs/datasets.md) for the transformation and reconstruction details.

## Synthetic speech

All speech in the 260 bundled evaluation clips under `generated_data/theory_of_time_v1/` was synthesized using OpenAI `gpt-4o-mini-tts`. These voices are AI-generated, not recordings of human speakers. See the [synthetic-audio notice](generated_data/theory_of_time_v1/README.md) for the included tracks and inventory.

Use of the generation service is governed by the applicable OpenAI agreement and policies. The [OpenAI Services Agreement](https://openai.com/policies/services-agreement/) describes rights and responsibilities for API inputs and outputs; it does not replace rights in third-party input material.

## Models, services, and software dependencies

Model configurations and archived outputs identify the evaluated systems; model weights and tokenizers are acquired separately. Provider services, external model implementations, and dependencies retain their respective licenses and terms. Dependency specifications in `requirements/` do not relicense those packages. See [model documentation](docs/models.md) for model sources, runtime specifications, and separately acquired implementations.

Archived outputs may reproduce source-dataset quotations or other third-party content. Inclusion in the results archive does not remove the terms applicable to that content.
