"""Findings and reports shared by every check."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

LEVELS = ("fail", "warn", "info")
_LEVEL_COLORS = {"fail": "1;31", "warn": "33", "info": "36"}

# The four harness details from the report, named after what they look like in a run.
IDLE_CLIP = "Idle Clip"                # check 1: PPO ratio taken against a recomputed policy
LOST_SEED = "Lost Seed"                # check 2: a data seed that never reached an arm
STUCK_BATCH = "Stuck Batch"            # check 3: updates that keep training on one batch
ROGUE_NORMALISER = "Rogue Normaliser"  # check 4: a loss normaliser that differs from its equation


@dataclass
class Finding:
    """One observation from a check.

    ``level`` is ``"fail"`` when the defining quantity contradicts the intended
    setup, ``"warn"`` when the setup cannot be confirmed from what was given, and
    ``"info"`` for measurements worth keeping next to the results. ``tag``
    names the harness detail a finding matches, when it matches one exactly.
    """

    check: int
    level: str
    title: str
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    tag: str = ""

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level!r}")


@dataclass
class Report:
    """A list of findings plus the measured statistics behind them."""

    findings: List[Finding] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    subject: str = ""

    def add(self, check: int, level: str, title: str, detail: str = "", tag: str = "", **evidence: Any) -> Finding:
        f = Finding(check, level, title, detail, dict(evidence), tag)
        self.findings.append(f)
        return f

    def extend(self, other: "Report", prefix: str = "") -> "Report":
        for f in other.findings:
            title = f"{prefix}{f.title}" if prefix else f.title
            self.findings.append(Finding(f.check, f.level, title, f.detail, dict(f.evidence), f.tag))
        if other.stats:
            key = prefix.rstrip(": ") or other.subject or f"part{len(self.stats)}"
            self.stats[key] = other.stats
        return self

    def by_level(self, level: str) -> List[Finding]:
        return [f for f in self.findings if f.level == level]

    @property
    def ok(self) -> bool:
        """True when no finding has level ``fail``."""
        return not self.by_level("fail")

    def raise_on_fail(self) -> None:
        if not self.ok:
            raise HarnessCheckError(str(self))

    def format(self, levels: Iterable[str] = LEVELS, color: bool = False) -> str:
        """Render the findings as text. ``color`` adds ANSI colours for terminals."""
        wanted = set(levels)

        def paint(text: str, code: str) -> str:
            return f"\x1b[{code}m{text}\x1b[0m" if color else text

        lines = []
        if self.subject:
            lines.append(paint(self.subject, "1"))
        order = {lvl: i for i, lvl in enumerate(LEVELS)}
        for f in sorted(self.findings, key=lambda f: (order[f.level], f.check)):
            if f.level not in wanted:
                continue
            tag = paint(f"[{f.level.upper():4}]", _LEVEL_COLORS[f.level])
            name = f" ({paint(f.tag, '1')})" if f.tag else ""
            lines.append(f"  {tag} check {f.check}{name}: {f.title}")
            if f.detail:
                for chunk in f.detail.splitlines():
                    lines.append(paint(f"         {chunk}", "2"))
        if len(lines) == (1 if self.subject else 0):
            hidden = [f for f in self.findings if f.level not in wanted]
            done = f"passed ({len(hidden)} info hidden)" if hidden else "no findings"
            lines.append("  " + paint(done, "32"))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


class HarnessCheckError(AssertionError):
    """Raised by :meth:`Report.raise_on_fail`."""
