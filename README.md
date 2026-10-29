# PicusLabMarine EURS Track 1 reproduction

Reproduces Testing submission **19433**, account/user **2102**.

1. After the public weights URL is live, run `python repro_check.py . --download-weights`. This downloads, verifies and extracts the single archive from `weights/weights_link.yaml`, recording the extraction for subsequent checker runs. `MODEL_INDEX.json` will be `weights/MODEL_INDEX.json`. Do not commit the extracted weights.
2. Build the root Dockerfile (network is needed at build time only).
3. Run inference without network, with read-only model assets, a writable output directory and `/tmp`:

```bash
python repro_check.py .
python repro_check.py . --run --input /path/to/degraded --n 5
python run.py --model-dir weights --input-dir /path/to/degraded --output-dir /path/to/fresh-output --device cuda
```

`run.main(model_dir, input_dir, output_dir, device)` is the entry point. Set `EURS_SAM_PYTHON` only when running outside Docker to point to the separate Torch2.10 SAM interpreter. The root Dockerfile sets this automatically. The output directory must be empty. Inference requires one NVIDIA GPU; both processes share it and fit within24GB by the observed allocation. Source inputs keep their original suffix `_waterType_level`; arbitrary scene prefixes are supported. Outputs are `restored/<stem>.jpg` (JPEG99) and `seg_masks/<stem>.png` (IDs0–4), longest side1024.

See `METHOD.md` for data use, architecture, exact precision/codec choices, licenses and the scope of local verification. The Docker build itself is not locally certified; fixed-image CUDA equivalence was checked against the actual uploaded archive using the original installed runtimes. Training is not rerun by this package.
