"""Print selected saved paper results as Markdown; Python standard library only."""
import json
from pathlib import Path


def table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(map(str, row)) + " |")
    print()


def main():
    data = json.loads((Path(__file__).resolve().parent / "data/results.json").read_text())
    print("# Selected paper results\n")
    print("Saved aggregate results; confidence intervals are stored, not recomputed.\n")
    print("## Paired contrasts\n")
    rows = []
    for contrast in data["contrasts"]:
        for model, value in contrast["per_model"].items():
            if value is not None:
                rows.append([contrast["regime"], contrast["contrast"], model,
                             f'{value["delta"]:+.2f}',
                             f'[{value["lo"]:+.2f}, {value["hi"]:+.2f}]', value["n_programs"]])
    table(["Setting", "Contrast", "Model", "Difference (pp)", "95% CI", "Programs"], rows)
    print("## Merge operator\n")
    print("Accuracy as a percentage of the untuned model's clean accuracy. "
          "Gated cells failed the format criterion; absent groups were not run.\n")
    rows = []
    for model, result in data["merge_operator"].items():
        for direction in ("forward", "backward"):
            for group, values in result.get(direction, {}).items():
                rows.append([model, direction, group] + [
                    "gated" if values[arm] is None else f"{values[arm]:.1f}"
                    for arm in ("base", "linear", "TIES", "DARE-TIES")])
    table(["Model", "Direction", "Condition", "Base", "Linear", "TIES", "DARE-TIES"], rows)
    print("## Coding retention (MBPP+)\n")
    print("Pass@1 on the extended tests, 399 tasks. †: at least half of replies contain no function.\n")
    arms = list(next(iter(data["retention"].values())))
    rows = []
    for model, values in data["retention"].items():
        rows.append([model] + [f'{100 * values[arm]["pass@1_plus"]:.1f}' +
                    ("†" if values[arm]["raw_no_def_rate"] >= 0.5 else "") for arm in arms])
    table(["Model"] + arms, rows)


if __name__ == "__main__":
    main()
