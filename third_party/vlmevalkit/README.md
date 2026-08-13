# VLMEvalKit Patch Bundle

This directory contains small IC-VCO compatibility patches for a user-managed
VLMEvalKit checkout. It intentionally does not vendor VLMEvalKit source code.

## Apply

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
git checkout 58fdeb6b980bda22096d912d70d1c858dedc84fd
bash /path/to/icvco/third_party/vlmevalkit/apply_patches.sh . check
bash /path/to/icvco/third_party/vlmevalkit/apply_patches.sh .
```

Use `check` first when applying to a newer upstream commit. The script warns if
the checkout is not at the tested commit and still runs `git apply --check`
before modifying files.

## Included Patches

- `0001-icvco-llava-loading.patch`: required for the LLaVA-NeXT and
  LLaVA-OneVision-HF evaluation path used by IC-VCO.
- `0002-icvco-crpe-path-resolution.patch`: resolves legacy CRPE image paths
  to `CRPE_EXIST` or `CRPE_RELATION` before VLMEvalKit parses image inputs.
- `0003-icvco-yorn-temperature-cap.patch`: caps Y/N judge retry temperature
  below DashScope's OpenAI-compatible API upper bound.
