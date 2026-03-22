# DSRF-HEAL Experiment Runner

This repository contains the code used to evaluate **DSRF (Diagnosis-Driven Selective Repair Framework)** for embodied goal generation on the `HEAL/` benchmark through OpenAI-compatible APIs.

The project supports:

- direct prompting
- naive multi-round self-reflection
- constraint-only repair
- violation-only repair
- the full diagnosis-driven selective repair pipeline used in the paper

At a high level, DSRF follows the paper's repair loop:

`Constraint Parsing -> Validation -> Diagnosis -> Routing -> Repair`

The implementation is built for large-scale benchmark runs and automatically chains generation, validation, and summarization.

## Project Overview

### Benchmark

The benchmark root is `HEAL/`. The current setup uses the full HEAL test set with 10 sub-splits:

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

### Repair Setting

The validator checks generated symbolic goals against explicit environment constraints and reports five primitive violation types:

- `O1`: nonexistent object in node goals
- `S`: illegal target state in node goals
- `O2`: nonexistent object in edge goals
- `R`: invalid relation type
- `O3`: invalid relation-target assignment

Primitive violations are then mapped to repair-oriented diagnoses such as:

- `entity grounding`
- `state property`
- `structural relation`
- `relation-target mismatch`
- `feasibility`

These diagnoses determine whether the pipeline performs:

- constrained regeneration
- localized structural repair
- conservative abstention

### Main Entry Points

Recommended scripts:

- `experiment_direct_prompting.py`: direct prompting
- `experiment_naive_self_reflection.py`: naive self-reflection with violation hints
- `experiment_constraint_only_repair.py`: one-round constraint-only repair
- `experiment_violation_only_repair.py`: one-round violation-only repair
- `experiment_diagnosis_guided_pipeline.py`: full DSRF pipeline

Compatibility wrappers are also included:

- `run.py`
- `control_pipeline.py`
- `pipeline.py`

### Repository Layout

```text
configs/        model definitions
HEAL/           benchmark files
src/            core implementation
outputs/        raw model outputs
validations/    validator outputs
results/        summary plots and metrics
```

## Supplementary Case Studies

The examples below are organized as qualitative evidence for the three routing behaviors discussed in the paper, plus one residual failure case.

### Case 1: Constrained Regeneration for Distractor-Induced Drift

**Task.** Collect aluminum cans and place them into a bucket.

**Environment constraints.** The scene contains six cans (`pop.n.02_*`) and one bucket. The cans are initially distributed across the bed and floor. The prompt licenses the cans as `Frozen`, but does not justify additional cleanliness edits such as `Stained` or `Dusty` as task goals.

**Initial output and primitive violations.** The raw prediction adds twelve unsupported unary goals such as `not Stained(pop)` and `not Dusty(pop)`, and also uses an invalid containment formulation. Validation reports heavy `S` and `O2` violations (`S=12`, `O2=3`).

**Diagnosis and routing.** DSRF identifies this as a broader repairable deviation rather than a local structural mismatch. The dominant diagnosis is `entity grounding`, so the sample is routed to **constrained regeneration**.

**Repair behavior.** The repair prompt re-injects explicit objects, valid states, and legal relation constraints. After two bounded repair rounds, the unsupported unary goals are removed and the result is reduced to the valid packing relations:

```json
{"node goals": [], "edge goals": [["inside", "pop.n.02_1", "bucket.n.01_1"], ["inside", "pop.n.02_2", "bucket.n.01_1"], ["inside", "pop.n.02_3", "bucket.n.01_1"], ["inside", "pop.n.02_4", "bucket.n.01_1"], ["inside", "pop.n.02_5", "bucket.n.01_1"], ["inside", "pop.n.02_6", "bucket.n.01_1"]]}
```

**Outcome.** All primitive violations are removed. This case matches the paper's claim that explicit constraints are especially helpful when distractors push the model toward unsupported object properties.

### Case 2: Localized Repair for Relation-Target Mismatch

**Task.** Load dirty clothes into the washing machine.

**Environment constraints.** The scene includes a `washing_machine` and dirtyable clothing objects such as `clothes_pants` and `clothes_shirt`. States such as `ON`, `CLOSED`, and `DIRTY` are valid, but the relation-target constraints do not accept the predicted `INSIDE(*, washing_machine)` edges in the final symbolic form.

**Initial output and primitive violations.** The initial prediction is largely correct at the node level:

```json
{"node goals": [{"name": "washing_machine", "state": "ON"}, {"name": "washing_machine", "state": "CLOSED"}, {"name": "clothes_pants", "state": "DIRTY"}, {"name": "clothes_shirt", "state": "DIRTY"}], "edge goals": [{"from_name": "clothes_pants", "relation": "INSIDE", "to_name": "washing_machine"}, {"from_name": "clothes_shirt", "relation": "INSIDE", "to_name": "washing_machine"}]}
```

Validation reports pure `O3` errors (`O3=2`), so the issue is concentrated on relation-target compatibility rather than global scene misunderstanding.

**Diagnosis and routing.** DSRF diagnoses `relation-target mismatch` and routes the sample to **localized repair** rather than full regeneration.

**Repair behavior.** The structural repair path edits only the edge layer. It preserves the valid washer and clothing state goals, removes the incompatible `INSIDE` edges, and avoids rewriting the entire output.

**Outcome.** `O3` drops from `2` to `0` in one repair round. This is the exact qualitative pattern emphasized in the paper: local structural repair reduces structural drift by correcting only the mismatched relation component.

### Case 3: Feasibility-Triggered Abstention

**Task.** Put groceries in the fridge.

**Environment constraints.** The scene contains objects such as `freezer`, `food_food`, `bedroom`, and `dining_room`, but the key task entities `fridge`, `bags`, and `groceries` are absent. This creates a direct scene-task inconsistency.

**Initial output and primitive violations.** The raw prediction hallucinates the missing entities and proposes goals such as opening a nonexistent fridge and interacting with nonexistent bags. Validation reports concentrated object-level errors (`O1=3`, `O2=2`).

**Diagnosis and routing.** Because the missing entities are central to task execution and object-level errors dominate, DSRF raises both `entity grounding` and `feasibility`. The sample is therefore routed to **abstain**.

**Repair behavior.** No open-ended rewriting is attempted. Following the paper's conservative-termination design, DSRF returns an empty goal output:

```json
{"node goals": [], "edge goals": []}
```

**Outcome.** The final output is hallucination-free and avoids constructing an executable-looking pseudo-goal outside the actual scene. This case illustrates why abstention is preferable when the task lies outside the executable space of the environment.

### Case 4: Residual Failure under Object Removal

**Task.** Install a modem.

**Environment constraints.** In this object-removal variant, the relevant scene no longer contains the key task object `modem.n.01_1`. The remaining context only provides support objects such as a table or floor, so the intended installation target is effectively unavailable.

**Initial output and primitive violations.** The initial prediction still hallucinates the removed modem:

```json
{"node goals": [["Toggled_On", "modem.n.01_1"]], "edge goals": [["Under", "modem.n.01_1", "table.n.02_1"]]}
```

Validation reports simultaneous object and state inconsistencies (`O1=1`, `S=1`, `O2=1`).

**Diagnosis and routing.** DSRF diagnoses `entity grounding + state property` and routes the case to **constrained regeneration**. This sample does not cross the framework's feasibility threshold, so abstention is not triggered.

**Repair behavior.** After two repair rounds, the invalid node goal is removed, but the regenerated edge is still incorrect:

```json
{"node goals": [], "edge goals": [["Under", "agent.n.01_1", "table.n.02_1"]]}
```

**Outcome.** The final output still has residual `O2=1`. This boundary case shows that generic constrained regeneration can still substitute the missing key entity with another scene object, and therefore motivates stronger feasibility gating in severe object-removal settings.

## Running the Project

### Environment

- Python `3.10`

Install dependencies:

```bash
pip install -r requirements.txt
```

### API Configuration

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

### Common Commands

Run the full HEAL benchmark with direct prompting:

```bash
python experiment_direct_prompting.py --llm qwen2.5-7b-instruct --heal_root HEAL --out_dir outputs
```

Run the full diagnosis-driven selective repair pipeline:

```bash
python experiment_diagnosis_guided_pipeline.py --llm qwen3-max --heal_root HEAL --out_dir outputs
```

Run a minimal smoke test on the first sample of `behavior/baseline`:

```bash
python experiment_direct_prompting.py --llm MiniMax-M2.5 --single_behavior_baseline
```

In full mode, each run automatically chains:

1. generation
2. validation
3. summarization

### Common Arguments

- `--config`: path to the LLM config file, default `configs/llms.yaml`
- `--heal_root`: benchmark root, default `HEAL`
- `--out_dir`: generation output root, default `outputs`
- `--validations_root`: validation output root, default `validations`
- `--results_dir`: summary figure output root, default `results`
- `--max_retries`: retry count for failed API calls, default `3`
- `--max_workers`: parallel worker count; `<= 0` means auto-expand to all chunks

CSV files are split into chunks of 100 samples for parallel execution. Workers write temporary partial files first, and the main process merges them in chunk order after completion.

### Output Layout

Each experiment creates a run directory under `outputs/`:

```text
outputs/{run_dir_name}/
```

The default naming pattern is:

```text
YYYYmmdd_HHMMSS__{llm_name}
```

Method-specific prefixes may also appear:

- `simpleReflect_{timestamp}__{llm}`
- `constraintOnlyRepair_{timestamp}__{llm}`
- `violationOnlyRepair_{timestamp}__{llm}`
- `taxonomyPipeline_{timestamp}__{llm}`

Each dataset is written to one JSONL file:

```text
{dataset_name}__{llm_name}.jsonl
```

The validation outputs mirror the same run structure under `validations/`, and summary figures are written under `results/`.

### Record Format

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

### Standalone Validation

You can validate an existing run directory independently:

```bash
python validate.py --output_run_dir outputs/20260306_230451__qwen --validations_root validations
```

## Notes

- This repository uses OpenAI-compatible APIs only; it does not depend on local inference backends.
- The `.env`, `outputs/`, `validations/`, and `results/` directories are intended to stay local and are ignored by Git.
