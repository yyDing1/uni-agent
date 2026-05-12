"""Offline tail-latency analyzer for Uni-Agent rollouts.

Reads per-rollout `interaction_result.json` files produced by `UniAgentLoop`
and produces:
  - <out_dir>/per_step.jsonl           one row per trajectory turn, all steps
  - <out_dir>/per_trajectory.jsonl     one row per trajectory
  - <out_dir>/tail_report.md           human-friendly Markdown report
                                       (top-K latest-finishing trajectories per step,
                                        with their time composition and timeout details)

Expected layout under --log-dir:
    <log_dir>/<batch_tag>/<run_id>/interaction_result.json   (multi-step mode)
or  <log_dir>/<run_id>/interaction_result.json               (legacy single-batch mode)

Usage:
    python examples/agent_interaction/analyze_tail.py \
        --log-dir /tmp/swebench_qwen3_coder \
        --top-k 10 \
        --out-dir ./tail_analysis
"""

# ruff: noqa: E501

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step_idx: int
    tool_name: str | None
    tool_time: float | None
    tool_outcome: str | None
    tool_recovery_time: float | None
    llm_time: float | None
    step_start_ts: float | None
    step_end_ts: float | None
    exit_reason: str
    action: str
    observation: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StepRecord":
        return cls(
            step_idx=d.get("step_idx", -1),
            tool_name=d.get("tool_name"),
            tool_time=d.get("tool_time"),
            tool_outcome=d.get("tool_outcome"),
            tool_recovery_time=d.get("tool_recovery_time"),
            llm_time=d.get("llm_time"),
            step_start_ts=d.get("step_start_ts"),
            step_end_ts=d.get("step_end_ts"),
            exit_reason=d.get("exit_reason", ""),
            action=d.get("action", ""),
            observation=d.get("observation", ""),
        )


@dataclass
class TrajectoryRecord:
    batch_tag: str
    run_id: str
    path: Path
    start_ts: float | None
    end_ts: float | None
    execution_time: float
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def num_turns(self) -> int:
        return len(self.steps)

    @property
    def sum_llm_time(self) -> float:
        return sum(s.llm_time or 0.0 for s in self.steps)

    @property
    def sum_tool_time(self) -> float:
        return sum(s.tool_time or 0.0 for s in self.steps)

    @property
    def sum_recovery_time(self) -> float:
        return sum(s.tool_recovery_time or 0.0 for s in self.steps)

    @property
    def timeout_steps(self) -> list[StepRecord]:
        return [s for s in self.steps if s.tool_outcome == "timeout"]

    @property
    def terminal_dead_steps(self) -> list[StepRecord]:
        return [s for s in self.steps if s.tool_outcome == "terminal_dead"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def discover_trajectory_files(log_dir: Path) -> list[tuple[str, Path]]:
    """Find every interaction_result.json under log_dir and tag it with batch_tag.

    Recognizes two layouts:
      - log_dir/<batch_tag>/<run_id>/interaction_result.json   -> batch_tag = parent.parent.name
      - log_dir/<run_id>/interaction_result.json               -> batch_tag = "_default"
    """
    out: list[tuple[str, Path]] = []
    for p in log_dir.rglob("interaction_result.json"):
        rel_parts = p.relative_to(log_dir).parts
        # rel_parts looks like ("step_000", "<uuid>", "interaction_result.json")
        # or ("<uuid>", "interaction_result.json")
        if len(rel_parts) == 3:
            out.append((rel_parts[0], p))
        elif len(rel_parts) == 2:
            out.append(("_default", p))
        else:
            logger.warning("Skipping unrecognized path layout: %s", p)
    return out


def load_trajectory(batch_tag: str, path: Path) -> TrajectoryRecord | None:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None

    run_id = path.parent.name
    steps = [StepRecord.from_dict(s) for s in data.get("trajectory", [])]
    return TrajectoryRecord(
        batch_tag=batch_tag,
        run_id=run_id,
        path=path,
        start_ts=data.get("start_ts"),
        end_ts=data.get("end_ts"),
        execution_time=float(data.get("execution_time", 0.0)),
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Per-batch tail analysis
# ---------------------------------------------------------------------------


def group_by_batch(trajectories: list[TrajectoryRecord]) -> dict[str, list[TrajectoryRecord]]:
    groups: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for t in trajectories:
        groups[t.batch_tag].append(t)
    return dict(sorted(groups.items()))


def pick_tail(trajectories: list[TrajectoryRecord], top_k: int) -> list[TrajectoryRecord]:
    """Latest-finishing trajectories in this batch (sorted by end_ts desc).

    end_ts is wall-clock so this is the right key for 'who held up the batch'.
    """
    valid = [t for t in trajectories if t.end_ts is not None]
    valid.sort(key=lambda t: t.end_ts, reverse=True)  # type: ignore[arg-type, return-value]
    return valid[:top_k]


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_jsonl(per_step_path: Path, per_traj_path: Path, trajectories: list[TrajectoryRecord]) -> None:
    with per_step_path.open("w", encoding="utf-8") as f_step, per_traj_path.open("w", encoding="utf-8") as f_traj:
        for traj in trajectories:
            f_traj.write(
                json.dumps(
                    {
                        "batch_tag": traj.batch_tag,
                        "run_id": traj.run_id,
                        "start_ts": traj.start_ts,
                        "end_ts": traj.end_ts,
                        "execution_time": traj.execution_time,
                        "num_turns": traj.num_turns,
                        "sum_llm_time": traj.sum_llm_time,
                        "sum_tool_time": traj.sum_tool_time,
                        "sum_recovery_time": traj.sum_recovery_time,
                        "num_timeouts": len(traj.timeout_steps),
                        "num_terminal_dead": len(traj.terminal_dead_steps),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for s in traj.steps:
                f_step.write(
                    json.dumps(
                        {
                            "batch_tag": traj.batch_tag,
                            "run_id": traj.run_id,
                            "step_idx": s.step_idx,
                            "step_start_ts": s.step_start_ts,
                            "step_end_ts": s.step_end_ts,
                            "llm_time": s.llm_time,
                            "tool_name": s.tool_name,
                            "tool_time": s.tool_time,
                            "tool_outcome": s.tool_outcome,
                            "tool_recovery_time": s.tool_recovery_time,
                            "exit_reason": s.exit_reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def _fmt_pct(numer: float, denom: float) -> str:
    if denom <= 0:
        return "n/a"
    return f"{numer / denom * 100:.1f}%"


def _fmt_time(t: float | None) -> str:
    if t is None:
        return "  n/a"
    return f"{t:6.2f}s"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... <{len(text) - max_chars} chars elided>"


def write_markdown_report(
    out_path: Path,
    groups: dict[str, list[TrajectoryRecord]],
    top_k: int,
    action_max_chars: int,
    observation_max_chars: int,
) -> None:
    lines: list[str] = []
    lines.append("# Uni-Agent Tail Latency Report\n")
    lines.append(f"Total batches: **{len(groups)}**\n")
    total_traj = sum(len(v) for v in groups.values())
    lines.append(f"Total trajectories: **{total_traj}**\n")

    # Overall summary table
    lines.append("\n## Per-batch summary\n")
    lines.append("| batch_tag | #traj | wall (s) | mean exec (s) | p90 exec | max exec | sum_tool | sum_recovery | #timeouts | #terminal_dead |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag, trajs in groups.items():
        if not trajs:
            continue
        wall_start = min((t.start_ts for t in trajs if t.start_ts is not None), default=None)
        wall_end = max((t.end_ts for t in trajs if t.end_ts is not None), default=None)
        wall = (wall_end - wall_start) if (wall_start is not None and wall_end is not None) else 0.0

        execs = sorted(t.execution_time for t in trajs)
        mean_exec = sum(execs) / len(execs) if execs else 0.0
        p90 = execs[int(len(execs) * 0.9)] if execs else 0.0
        max_exec = execs[-1] if execs else 0.0

        sum_tool = sum(t.sum_tool_time for t in trajs)
        sum_recov = sum(t.sum_recovery_time for t in trajs)
        n_to = sum(len(t.timeout_steps) for t in trajs)
        n_td = sum(len(t.terminal_dead_steps) for t in trajs)
        lines.append(
            f"| `{tag}` | {len(trajs)} | {wall:.1f} | {mean_exec:.1f} | {p90:.1f} | {max_exec:.1f} | "
            f"{sum_tool:.1f} | {sum_recov:.1f} | {n_to} | {n_td} |"
        )

    # Detailed tail per batch
    for tag, trajs in groups.items():
        lines.append(f"\n---\n\n## Batch `{tag}` — top {top_k} latest-finishing trajectories\n")
        tail = pick_tail(trajs, top_k)
        if not tail:
            lines.append("_(no trajectories with end_ts)_")
            continue

        # Reference wall-start for this batch (earliest start) to compute relative times
        batch_t0 = min((t.start_ts for t in trajs if t.start_ts is not None), default=None)

        for rank, traj in enumerate(tail, start=1):
            rel_end = (traj.end_ts - batch_t0) if (traj.end_ts is not None and batch_t0 is not None) else None
            rel_start = (traj.start_ts - batch_t0) if (traj.start_ts is not None and batch_t0 is not None) else None
            lines.append(f"### #{rank} `{traj.run_id}`\n")
            lines.append(f"- file: `{traj.path}`")
            lines.append(
                f"- start (rel to batch): {rel_start:.1f}s, end (rel): {rel_end:.1f}s"
                if rel_start is not None and rel_end is not None
                else "- start/end relative time: n/a"
            )
            lines.append(
                f"- execution_time: **{traj.execution_time:.1f}s** | num_turns: {traj.num_turns}"
            )
            lines.append(
                f"- sum_llm: {traj.sum_llm_time:.1f}s ({_fmt_pct(traj.sum_llm_time, traj.execution_time)}) | "
                f"sum_tool: {traj.sum_tool_time:.1f}s ({_fmt_pct(traj.sum_tool_time, traj.execution_time)}) | "
                f"sum_recovery: {traj.sum_recovery_time:.1f}s ({_fmt_pct(traj.sum_recovery_time, traj.execution_time)})"
            )
            lines.append(
                f"- #timeouts: **{len(traj.timeout_steps)}** | #terminal_dead: **{len(traj.terminal_dead_steps)}**\n"
            )

            # Per-turn table
            lines.append("| step | tool | llm | tool_time | outcome | recovery | exit_reason |")
            lines.append("|---:|---|---:|---:|---|---:|---|")
            for s in traj.steps:
                lines.append(
                    f"| {s.step_idx} | `{s.tool_name or '-'}` | {_fmt_time(s.llm_time)} | "
                    f"{_fmt_time(s.tool_time)} | {s.tool_outcome or '-'} | "
                    f"{_fmt_time(s.tool_recovery_time)} | {s.exit_reason} |"
                )

            # Highlight timeout / terminal_dead steps with their commands
            bad_steps = [s for s in traj.steps if s.tool_outcome in ("timeout", "terminal_dead")]
            if bad_steps:
                lines.append("\n**Timeout / terminal-dead invocations:**\n")
                for s in bad_steps:
                    lines.append(
                        f"- step **{s.step_idx}** ({s.tool_name}, outcome=`{s.tool_outcome}`, "
                        f"exec={_fmt_time(s.tool_time).strip()}, recovery={_fmt_time(s.tool_recovery_time).strip()}):"
                    )
                    lines.append("\n```bash")
                    lines.append(_truncate(s.action.strip(), action_max_chars))
                    lines.append("```")
                    if s.observation:
                        lines.append("Observation (truncated):\n")
                        lines.append("```text")
                        lines.append(_truncate(s.observation.strip(), observation_max_chars))
                        lines.append("```")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Tail-latency analyzer for Uni-Agent rollouts")
    parser.add_argument("--log-dir", type=Path, required=True, help="Root log dir (UniAgentLoop log_dir).")
    parser.add_argument("--out-dir", type=Path, default=Path("./tail_analysis"), help="Where to write reports.")
    parser.add_argument("--top-k", type=int, default=10, help="How many latest-finishing trajectories per batch.")
    parser.add_argument("--action-chars", type=int, default=2000, help="Max chars of action command to embed.")
    parser.add_argument("--obs-chars", type=int, default=1000, help="Max chars of observation to embed.")
    args = parser.parse_args()

    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.is_dir():
        logger.error("log-dir %s does not exist", log_dir)
        return 2

    paths = discover_trajectory_files(log_dir)
    logger.info("Found %d trajectory files under %s", len(paths), log_dir)
    if not paths:
        return 1

    trajectories: list[TrajectoryRecord] = []
    for batch_tag, path in paths:
        traj = load_trajectory(batch_tag, path)
        if traj is not None:
            trajectories.append(traj)
    logger.info("Loaded %d trajectories successfully", len(trajectories))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_step_path = args.out_dir / "per_step.jsonl"
    per_traj_path = args.out_dir / "per_trajectory.jsonl"
    report_path = args.out_dir / "tail_report.md"

    write_jsonl(per_step_path, per_traj_path, trajectories)
    logger.info("Wrote %s and %s", per_step_path, per_traj_path)

    groups = group_by_batch(trajectories)
    write_markdown_report(
        report_path,
        groups,
        top_k=args.top_k,
        action_max_chars=args.action_chars,
        observation_max_chars=args.obs_chars,
    )
    logger.info("Wrote %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
