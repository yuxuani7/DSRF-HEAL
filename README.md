# HEAL Multi-Model Experiment Runner

This project runs multiple LLM-based goal-generation experiments on the `HEAL/` benchmark through OpenAI-compatible APIs. It supports direct prompting, naive self-reflection, controlled repair baselines, and the diagnosis-driven selective repair pipeline used in the paper.

## Environment

- Python `3.10`

Install dependencies:

```bash
pip install -r requirements.txt
```

## API Configuration

Create a local `.env` file in the project root:

```env
ALI_ACCESS_TOKEN=your_real_token
MODELSCOPE_ACCESS_TOKEN=your_real_token
```

Model definitions are stored in `configs/llms.yaml`. The config loader supports `${ENV_VAR}` placeholders for `api_key`.

Each model entry includes:

- `name`: alias passed through `--llm`
- `provider`: currently fixed to `openai_compatible`
- `base_url`: OpenAI-compatible endpoint
- `api_key`: raw key or `${ENV_VAR}`
- `model`: upstream model identifier
- `default_params`: default sampling parameters forwarded to `chat.completions.create(...)`

Current default sampling settings are:

```yaml
temperature: 0.2
top_p: 0.9
stream: true
```

## Dataset

The benchmark root is `HEAL/`. The current experiments use the full HEAL test set with 10 sub-splits:

- `behavior/baseline.csv`
- `behavior/distractor_injection.csv`
- `behavior/object_removal.csv`
- `behavior/scene_object_synonymous.csv`
- `behavior/scene_task_contradiction.csv`
- `virtualhome/baseline.csv`
- `virtualhome/distractor_injection.csv`
- `virtualhome/object_removal.csv`
- `virtualhome/scene_object_synonymous.csv`
- `virtualhome/scene_task_contradiction.csv`

The loader normalizes each sample into a shared structure with `dataset_name`, `task_id`, `prompt`, `messages`, and `meta`.

## Experiment Scripts

Recommended entry points:

- `experiment_direct_prompting.py`: direct prompting
- `experiment_naive_self_reflection.py`: naive self-reflection with violation hints
- `experiment_constraint_only_repair.py`: one-round constraint-only repair
- `experiment_violation_only_repair.py`: one-round violation-only repair
- `experiment_diagnosis_guided_pipeline.py`: diagnosis-driven selective repair pipeline

Compatibility wrappers are also kept:

- `run.py`
- `control_pipeline.py`
- `pipeline.py`

## Basic Usage

Run the full HEAL benchmark with direct prompting:

```bash
conda activate icss
python experiment_direct_prompting.py --llm qwen2.5-7b-instruct --heal_root HEAL --out_dir outputs
```

Run the diagnosis-driven pipeline:

```bash
conda activate icss
python experiment_diagnosis_guided_pipeline.py --llm qwen3-max --heal_root HEAL --out_dir outputs
```

Run a minimal smoke test on the first sample of `behavior/baseline`:

```bash
conda activate icss
python experiment_direct_prompting.py --llm MiniMax-M2.5 --single_behavior_baseline
```

In full mode, the pipeline automatically chains:

1. generation
2. validation
3. summarization

## Common Arguments

- `--config`: path to the LLM config file, default `configs/llms.yaml`
- `--heal_root`: benchmark root, default `HEAL`
- `--out_dir`: generation output root, default `outputs`
- `--validations_root`: validation output root, default `validations`
- `--results_dir`: summary figure output root, default `results`
- `--max_retries`: retry count for failed API calls, default `3`
- `--max_workers`: parallel worker count; `<= 0` means auto-expand to all chunks

CSV files are split into chunks of 100 samples for parallel execution. Workers write temporary partial files first, and the main process merges them in chunk order after completion.

## Output Layout

Each experiment creates a run directory under `outputs/`:

```text
outputs/{run_dir_name}/
```

The naming pattern is:

```text
YYYYmmdd_HHMMSS__{llm_name}
```

Method-specific prefixes may be used, for example:

- `simpleReflect_{timestamp}__{llm}`
- `constraintOnlyRepair_{timestamp}__{llm}`
- `violationOnlyRepair_{timestamp}__{llm}`
- `taxonomyPipeline_{timestamp}__{llm}`

Each dataset is written to one JSONL file:

```text
{dataset_name}__{llm_name}.jsonl
```

The validation outputs mirror the same run structure under `validations/`, and summary figures are written under `results/`.

## Record Format

Each output JSONL row contains:

- `run_id`
- `timestamp`
- `dataset`
- `task_id`
- `input`
- `llm`
- `output`
- `metrics`
- `error`
- `retry_errors`
- `meta`

Validation files additionally include hallucination judgments, primitive violation counts, empty-goal indicators, and parsing notes.

## Validator

The validator parses the prompt and checks the final goals against explicit environment constraints. It reports five primitive violation types:

- `O1`: nonexistent object in node goals
- `S`: illegal target state in node goals
- `O2`: nonexistent object in edge goals
- `R`: invalid relation type
- `O3`: invalid relation-target assignment

A sample is marked hallucinated if any of these violation counts is greater than zero.

Standalone validation on an existing run:

```bash
conda activate icss
python validate.py --output_run_dir outputs/20260306_230451__qwen --validations_root validations
```

## Project Structure

```text
configs/        model definitions
HEAL/           benchmark files
src/            core implementation
outputs/        raw model outputs
validations/    validator outputs
results/        summary plots and metrics
```

## Notes

- This repository uses OpenAI-compatible APIs only; it does not depend on local inference backends.
- The `.env`, `outputs/`, `validations/`, and `results/` directories are intended to stay local and are ignored by Git.
