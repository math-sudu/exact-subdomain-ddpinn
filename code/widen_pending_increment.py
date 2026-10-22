"""Add neural features to a pending increment while preserving its initial field."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from pilot.spectral_mixed import load_orientation_model, make_causal_model, freeze_history
from run_orientation_extension import save


@torch.no_grad()
def widen(saved, width, seed=1729):
    step = saved["model_steps"]
    if step != saved["completed_steps"] + 1:
        raise ValueError("Widen only the current pending increment")
    updated = deepcopy(saved)
    settings = updated["settings"]
    hidden = list(settings.get("increment_hidden", {}).get(str(step), settings["hidden"]))
    previous_width = hidden[-1]
    if width <= previous_width:
        raise ValueError("The new final hidden layer must be wider")
    hidden[-1] = width
    settings.setdefault("increment_hidden", {})[str(step)] = hidden
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = make_causal_model(settings, saved["c"], saved["case"])
        for _ in range(step):
            model.append_increment(copy_previous=False)
    state = model.state_dict()
    layer = 2 * (len(hidden) - 1)
    prefix = f"increments.{step - 1}."
    expanded_rows = {prefix + f"{kind}_trunk.{layer}.{parameter}"
                     for kind in ("kinematic", "stress") for parameter in ("weight", "bias")}
    expanded_columns = {prefix + f"{kind}_head.weight" for kind in ("kinematic", "stress")}
    for name, value in saved["model"].items():
        if name in expanded_rows:
            state[name][:previous_width].copy_(value)
        elif name in expanded_columns:
            state[name].zero_()
            state[name][:, :previous_width].copy_(value)
        else:
            state[name].copy_(value)
    model.load_state_dict(state)
    freeze_history(model)
    updated["model"] = state
    updated["trust_region_state"]["inner"] = {}
    updated.setdefault("network_widening", []).append(dict(
        step=step, evaluations=saved["evaluations"], previous_width=previous_width,
        width=width, seed=seed,
        initialization="Existing features retained; new hidden features have zero output weights."))
    return updated, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    _, saved, before = load_orientation_model(args.checkpoint)
    updated, after = widen(saved, args.width, args.seed)
    for name, value in saved["model"].items():
        if not name.startswith(f"increments.{saved['model_steps'] - 1}."):
            torch.testing.assert_close(updated["model"][name], value, atol=0, rtol=0)
    error = float((before.increments[-1].coefficients() -
                   after.increments[-1].coefficients()).abs().max())
    torch.testing.assert_close(after.increments[-1].coefficients(),
                               before.increments[-1].coefficients(), atol=1e-13, rtol=1e-12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, updated)
    print(json.dumps(dict(step=saved["model_steps"], width=args.width,
                         maximum_coefficient_change=error, prefix_unchanged=True)), flush=True)


if __name__ == "__main__":
    main()
