# Merging obfuscation specialists

Minimal companion code, program examples, and selected aggregate results for the paper.

## Data

The compressed JSONL files contain actual code, inputs and reference outputs:

| File in `data/` | Examples | Programs |
| --- | ---: | ---: |
| `train_python.jsonl.gz` | 26,841 | 1,563 |
| `val_python.jsonl.gz` | 1,917 | 111 |
| `eval_python.jsonl.gz` | 21,696 | 557 |
| `eval_javascript.jsonl.gz` | 7,044 | 168 |
| `eval_python_regenerated.jsonl.gz` | 15,720 | 557 |

Training and validation contain the six specialist conditions: L0, L1b, L1r,
L2, S1 and S2. The recovered evaluation files include these plus S3, S4 and six
depth-2 stacks. The regenerated file adds X1, X1m, X1s, six encoding-family
stacks and four seen depth-3/4 stacks. Read both Python evaluation files for
the full packaged condition set; their item IDs do not overlap.
Each row retains `item_id`, `program_id`, `condition`, `language`, `code`,
`entry_point`, `args_repr`, and `output_repr`. `args_repr` is the argument
representation, not JSON; `output_repr` is the canonical reference output.
Training rows also carry `split`; evaluation rows have their original metadata.
The same program appears under multiple conditions and inputs, so example counts
are larger than program counts. Transformation coverage varies.

```python
import gzip
import json

with gzip.open("data/train_python.jsonl.gz", "rt", encoding="utf-8") as f:
    clean_examples = [json.loads(line) for line in f]
clean_examples = [row for row in clean_examples if row["condition"] == "L0"]
```

The Python sources are APPS (`codeparrot/apps`), CRUXEval
(`cruxeval-org/cruxeval`) and HumanEval (`openai_humaneval`); original source
identifiers are retained in the program IDs. JavaScript source identifiers are
likewise retained. These are derived benchmark examples, not newly licensed
datasets; upstream attribution and terms still apply.

`splits.json` records the original program assignments (seed 17).
`manifest.json` records coverage and SHA-256 hashes. Exported Python training,
validation and evaluation programs are disjoint. Code, inputs, gold outputs and
row fields were preserved; test rows present in the original pairs files were
excluded from the exported training and validation files.

The later X1-family and depth-3/4 files could not be recovered and were rebuilt
from the same held-out parent programs using the repository's original
generators, resolved configurations and seed 17. Each included variant passed
the original gate: parsing, transformation checks, size limits, execution parity
against its parent on all recorded cases and gate inputs (8–23 cases per
program), reference-output checks, and runtime limits. Only the first three
recorded input cases are exported, following the original evaluation procedure.

These are **regenerated evaluation inputs**, not verified copies of the original
experiment files. X1/X1m/X1s coverage matches the paper (405/351/246 programs).
Some stacks can differ because the runtime gate depends on the machine.
`regeneration.json` records coverage, excluded program IDs, failed checks,
generator/source hashes, configuration, seed and Python/package versions.
The original saved result summaries were not recomputed on these regenerated
inputs. H1 quarantine and the legacy human-study test sets are not included.

## View results

Requires Python 3.10+; no packages, models, or GPU needed.

```bash
python results.py > report.md
```

This displays saved paired comparisons, the merge-operator ablation, and MBPP+
coding retention. Differences are percentage points; operator scores are relative
to the untuned clean-code baseline; retention scores are percentages.
Missing/gated comparisons are not treated as zero.

`data/results.json` contains existing experiment summaries. The contrasts and
operator results are preserved from the original analysis outputs; retention
contains only task counts, pass@1, and no-function rates from the original MBPP+
summaries. The paired intervals were computed with 2,000 program-clustered
bootstrap samples and are displayed as stored.

## Merge trained specialists

```bash
python -m pip install -r requirements.txt
python merge.py --base-model MODEL_ID --adapters adapters --out merged
```

Supply six trained adapters under `adapters/L0/`, `adapters/L1b/`,
`adapters/L1r/`, `adapters/L2/`, `adapters/S1/`, and `adapters/S2/`.
They must come from the same base model and use matching LoRA settings.
The paper uses rank 32, alpha 64, uniform weights, density 0.5 and seed 17.
The script calls PEFT's factor-space DARE-TIES merge, with `total` sign election,
and exports one adapter. Use `--operator ties` or `--operator linear` for the
operator comparisons. The base model is loaded in float32 on CPU; enough RAM
for the full model and adapters is required.

