# Deployment: GitHub → Hugging Face

```
push / PR ──► ci.yml       lint · types · tests (incl. PyTorch) · smoke pipeline + gate · Docker build
manual    ──► train.yml    full real-data run → artifact (nothing published)
tag v*    ──► release.yml  retrain from scratch → gate ──► model bundle → HF Hub (tagged)
                                                    ├──► image → ghcr.io
                                                    └──► app   → HF Space (pinned to that model tag)
```

## One-time setup

1. **Lockfile.** Run `make lock` and commit `uv.lock`. CI uses `--locked`, so a stale lock
   fails the build.
2. **Hugging Face.** Create a write token at huggingface.co → Settings → Access Tokens.
3. **GitHub repository settings.**
   - Secret `HF_TOKEN`: the token from step 2.
   - Variable `HF_MODEL_REPO`, e.g. `your-org/parpi-resistance`.
   - Variable `HF_SPACE_REPO`, e.g. `your-org/parpi-resistance-app`.
   - Environment `huggingface`. Add required reviewers if a human should approve each
     deployment.
4. **Names (done).** GitHub `sachitsatyal/parpi-resistance`; Hugging Face model
   `sawcheet/parpi-resistance`, Space `sawcheet/parpi-resistance-app`.

## Releasing

```bash
git tag v0.1.0 && git push origin v0.1.0
```

The release job retrains from the raw GEO files rather than reusing an earlier artifact. The
published model is therefore reproducible from the tagged commit alone. If `parpi gate` fails,
nothing is published.

The Space runs the same Dockerfile as the GHCR image. Its `PARPI_MODEL` variable is pinned to
`hf://<model repo>@<tag>`, so the app and the model card always describe the same model.

## Running locally

```bash
make docker
docker run --rm -p 7860:7860 -v $PWD/outputs:/models \
  -e PARPI_MODEL=/models/models/ridge parpi-resistance:dev
```
