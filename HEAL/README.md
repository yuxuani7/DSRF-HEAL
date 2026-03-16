---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
configs:
# Separate config for baseline data (has: task_id, task_name, original_natural_language_description, baseline_prompts)
- config_name: baseline
  data_files:
  - split: behavior
    path: "behavior/baseline.csv"
  - split: virtualhome
    path: "virtualhome/baseline.csv"

# Separate config for distractor injection data (has: task_id, task_name, distractor_injected_task, distractor, modified_prompts)
- config_name: distractor_injection
  data_files:
  - split: behavior
    path: "behavior/distractor_injection.csv"
  - split: virtualhome
    path: "virtualhome/distractor_injection.csv"

# Separate config for object removal data
- config_name: object_removal
  data_files:
  - split: behavior
    path: "behavior/object_removal.csv"
  - split: virtualhome
    path: "virtualhome/object_removal.csv"

# Separate config for scene object synonymous data
- config_name: scene_object_synonymous
  data_files:
  - split: behavior
    path: "behavior/scene_object_synonymous.csv"
  - split: virtualhome
    path: "virtualhome/scene_object_synonymous.csv"

# Separate config for scene task contradiction data
- config_name: scene_task_contradiction
  data_files:
  - split: behavior
    path: "behavior/scene_task_contradiction.csv"
  - split: virtualhome
    path: "virtualhome/scene_task_contradiction.csv"
---

