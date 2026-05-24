# Pretrained Text-to-Speech Evaluation

This project compares three pretrained audio generation models on the same expressive speech prompt set:

- `AudioLDM2`
- `Bark`
- `TangoFlux`

The goal is to compare prompt/style alignment and speech naturalness using generated `.wav` files.

## Files

- `prompts.json` - 50 evaluation prompts with transcript and style descriptions.
- `audio_generation.ipynb` - generates audio files for each model.
- `evaluate.ipynb` - evaluates existing generated audio without rerunning models.
- `tts_utils.py` - shared audio metric helpers.
- `requirements.txt` - Python dependencies.

## Output Structure

Generated audio is expected in:

```text
outputs/{model_name}/{prompt_id}.wav
```

Example:

```text
outputs/Bark/ancient_wizard.wav
```

Evaluation files are saved to:

```text
outputs/evaluation/
```

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate audio:

```text
Run audio_generation.ipynb
```

Evaluate existing audio:

```text
Run evaluate.ipynb
```

## Metrics

The evaluation computes:

- signal metrics: duration, loudness, silence ratio, spectral centroid, spectral rolloff, ZCR, temporal variance
- CLAP text-audio similarity
- UTMOS predicted speech naturalness

