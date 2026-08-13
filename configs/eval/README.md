# Eval Configs

This directory holds VLMEvalKit config-mode templates and repo-local examples.

Guidelines:

- Put model, dataset, and generation kwargs in config.
- Keep workflow controls such as `--work-dir`, `--mode`, `--reuse`, and
  `--judge` on the CLI.
- Do not combine `--config` with `--data` or `--model`.
- Use the model-specific single-node evaluation launchers under `scripts/`.

Templates:

- `llava_next_interleave_template.json`
- `llava_onevision_template.json`

These templates are starting points. Replace model paths and dataset names, and
verify wrapper/backend support before assuming each generation kwarg is active.

For LLaVA-NeXT, LLaVA-OneVision-HF, and CRPE evaluation, apply the VLMEvalKit
patch bundle under `third_party/vlmevalkit/` before running these configs.
