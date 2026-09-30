"""Merge the six paper specialists using PEFT's factor-space DARE-TIES."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

CONDITIONS = ("L0", "L1b", "L1r", "L2", "S1", "S2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Model ID or local model directory")
    parser.add_argument("--adapters", type=Path, required=True,
                        help="Directory containing L0/, L1b/, L1r/, L2/, S1/, S2/")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--operator", choices=("dare_ties", "ties", "linear"), default="dare_ties")
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not 0 < args.density <= 1:
        parser.error("density must be in (0, 1]")
    if args.out.exists():
        parser.error("output directory already exists; choose a new directory")
    paths = [args.adapters / name for name in CONDITIONS]
    configs = []
    for path in paths:
        if not (path / "adapter_config.json").is_file():
            parser.error(f"missing adapter_config.json in {path}")
        configs.append(json.loads((path / "adapter_config.json").read_text()))
    fields = ("r", "lora_alpha", "target_modules", "rank_pattern", "alpha_pattern",
              "use_rslora", "use_dora", "fan_in_fan_out", "modules_to_save", "bias")
    for config in configs:
        if config.get("peft_type") != "LORA":
            parser.error("all inputs must be LoRA adapters")
        if any(config.get(key) != configs[0].get(key) for key in fields):
            parser.error("specialists must have matching rank, scaling and target modules")

    import os
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, set_seed

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Match the original merge: float32 on CPU, uniform weights, total sign election.
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, device_map=None)
    model = PeftModel.from_pretrained(base, str(paths[0]), adapter_name=CONDITIONS[0])
    for name, path in zip(CONDITIONS[1:], paths[1:]):
        model.load_adapter(str(path), adapter_name=name)
    kwargs = dict(adapters=list(CONDITIONS), weights=[1 / 6] * 6,
                  adapter_name="merged", combination_type=args.operator)
    if args.operator != "linear":
        kwargs.update(density=args.density, majority_sign_method="total")
    model.base_model.add_weighted_adapter(**kwargs)
    model.set_adapter("merged")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        model.save_pretrained(tmp, selected_adapters=["merged"])
        source = Path(tmp) / "merged"
        if not source.is_dir():
            source = Path(tmp)
        args.out.mkdir()
        # PEFT-generated model cards can contain local paths; export only adapter files.
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copyfile(source / name, args.out / name)
    (args.out / "merge_settings.json").write_text(json.dumps({
        "conditions": CONDITIONS, "weights": [1 / 6] * 6,
        "operator": args.operator, "density": args.density if args.operator != "linear" else None,
        "seed": args.seed, "dtype": "float32", "majority_sign_method": "total",
    }, indent=2) + "\n")
    print(f"Saved merged adapter to {args.out}")


if __name__ == "__main__":
    main()
