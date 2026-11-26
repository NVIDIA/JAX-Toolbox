#!/usr/bin/env python3
"""Compare unprofiled results from run_experiment.py; never mix profile timings in.

Aggregation is descriptive: equal-weight mean of independent process means,
with paired deltas printed separately. No confidence/significance claim is made.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def read_measurement(model, raw):
    if raw.get("profiled", False):
        raise ValueError("Refusing a raw result marked profiled")
    if model == "gemma":
        # MaxText provides a loop average, NOT a list of per-step samples.
        mean = float(raw["autoregressive"]["step_in_ms"])
        samples = None
        signature = {"global_batch_size": raw["autoregressive"].get("global_batch_size")}
    elif model == "timesfm":
        samples = [float(x) for x in raw["samples_ms"]]
        mean = statistics.fmean(samples)
        signature = {key: raw.get(key) for key in (
            "level", "aot_compiled", "batch_size", "context_len", "horizon_len", "dtype", "random_weights")}
    else:
        samples = [float(x) * 1000 for x in raw["warm_execute_times_s"]]
        mean = statistics.fmean(samples)
        signature = {key: raw.get(key) for key in (
            "tokens_used", "num_recycles", "num_diffusion_samples", "flash_attention_implementation")}
    values = samples if samples is not None else [mean]
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("All recorded latencies must be finite and positive")
    return dict(mean_ms=mean, median_ms=statistics.median(samples) if samples else None,
                sample_count=len(samples) if samples else None, signature=signature)


def reduction(off, on):
    return (off - on) / off * 100


def summarize(manifest_path):
    manifest_path = Path(manifest_path)
    plan = json.loads(manifest_path.read_text())
    if plan["mode"] != "benchmark" or plan.get("profiled", True):
        raise ValueError("Refusing profiled timings: use a benchmark-mode manifest")
    records, grouped = [], {}
    reference_signature = None
    for job in plan["jobs"]:
        if job["status"] != "complete":
            raise ValueError("The experiment contains incomplete/failed arms")
        # Resolve relative to the manifest so copied result directories remain usable.
        relative = Path(job["raw_json"]).relative_to(Path(plan["out_dir"]))
        raw = json.loads((manifest_path.parent / relative).read_text())
        measurement = read_measurement(plan["model"], raw)
        signature = (measurement["signature"], measurement["sample_count"])
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise ValueError("Shapes, execution path, or sample counts differ across arms")
        record = dict(repeat=job["repeat"], arm=job["arm"], **measurement)
        records.append(record)
        key = (job["repeat"], job["arm"])
        if key in grouped:
            raise ValueError(f"Duplicate result: {key}")
        grouped[key] = record
    pairs = []
    for repeat in sorted({r["repeat"] for r in records}):
        off = grouped[(repeat, "cg_on_pdl_off")]
        on = grouped[(repeat, "cg_on_pdl_on")]
        pairs.append(dict(repeat=repeat, off_ms=off["mean_ms"], on_ms=on["mean_ms"],
                          marginal_pdl_reduction_pct=reduction(off["mean_ms"], on["mean_ms"]),
                          median_reduction_pct=(reduction(off["median_ms"], on["median_ms"])
                                                if off["median_ms"] is not None else None)))
    arms = {}
    for arm in sorted({r["arm"] for r in records}):
        means = [r["mean_ms"] for r in records if r["arm"] == arm]
        arms[arm] = dict(mean_ms=statistics.fmean(means),
                         between_process_stdev_ms=statistics.stdev(means) if len(means) > 1 else None,
                         process_count=len(means))
    off_ms, on_ms = arms["cg_on_pdl_off"]["mean_ms"], arms["cg_on_pdl_on"]["mean_ms"]
    return dict(model=plan["model"], settings=plan["settings"], profiled=False,
                aggregation="Equal-weight mean of process means; descriptive, not a significance test",
                arms=arms, pairs=pairs, records=records,
                marginal_pdl_reduction_pct=reduction(off_ms, on_ms))


def render(summary):
    lines = [f"\n{summary['model']}: unprofiled latency (milliseconds)",
             "arm                          process mean     process count"]
    for arm, values in summary["arms"].items():
        lines.append(f"{arm:28} {values['mean_ms']:12.4f} {values['process_count']:17d}")
    lines.append(f"Marginal PDL latency reduction, graphs ON: {summary['marginal_pdl_reduction_pct']:+.3f}%")
    for pair in summary["pairs"]:
        line = (f"  Pair {pair['repeat']}: {pair['off_ms']:.4f} -> {pair['on_ms']:.4f} ms "
                f"({pair['marginal_pdl_reduction_pct']:+.3f}%)")
        if pair["median_reduction_pct"] is not None:
            line += f"; per-arm median comparison {pair['median_reduction_pct']:+.3f}%"
        lines.append(line)
    lines.extend(["Positive = lower latency with PDL. Graphs-OFF rows, if present, are secondary.",
                  "These are descriptive measurements; small deltas may be noise."])
    if summary["model"] == "gemma":
        lines.append("Gemma: native MaxText synchronized-loop averages; no per-step samples or medians.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        print(render(summarize(args.manifest)))
    except (ValueError, KeyError, OSError, statistics.StatisticsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
