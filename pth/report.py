"""Findings and reports shared by every check."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

LEVELS = ("fail", "warn", "info")


@dataclass
class Finding:
    """One observation from a check.

    ``level`` is ``"fail"`` when the defining quantity contradicts the intended
    setup, ``"warn"`` when the setup cannot be confirmed from what was given, and
    ``"info"`` for measurements worth keeping next to the results.
    """

    check: int
    level: str
    title: str
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level!r}")


@dataclass
class Report:
    """A list of findings plus the measured statistics behind them."""

    findings: List[Finding] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    subject: str = ""

    def add(self, check: int, level: str, title: str, detail: str = "", **evidence: Any) -> Finding:
        f = Finding(check, level, title, detail, dict(evidence))
        self.findings.append(f)
        return f

    def extend(self, other: "Report", prefix: str = "") -> "Report":
        for f in other.findings:
            title = f"{prefix}{f.title}" if prefix else f.title
            self.findings.append(Finding(f.check, f.level, title, f.detail, dict(f.evidence)))
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

    def format(self, levels: Iterable[str] = LEVELS) -> str:
        wanted = set(levels)
        lines = []
        if self.subject:
            lines.append(self.subject)
        order = {lvl: i for i, lvl in enumerate(LEVELS)}
        for f in sorted(self.findings, key=lambda f: (order[f.level], f.check)):
            if f.level not in wanted:
                continue
            lines.append(f"  [{f.level.upper():4}] check {f.check}: {f.title}")
            if f.detail:
                for chunk in f.detail.splitlines():
                    lines.append(f"         {chunk}")
        if len(lines) == (1 if self.subject else 0):
            hidden = [f for f in self.findings if f.level not in wanted]
            lines.append(f"  passed ({len(hidden)} info hidden)" if hidden else "  no findings")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


class HarnessCheckError(AssertionError):
    """Raised by :meth:`Report.raise_on_fail`."""
