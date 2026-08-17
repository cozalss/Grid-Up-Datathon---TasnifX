"""Tracked dosya ve Git diff gecmisinde gitleaks-benzeri sir taramasi."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

PATTERNS = (
    re.compile(
        r"(?i)(?:api[_-]?key|secret|token|password|passwd|kaggle_key|epias_password)"
        r"\s*[:=]\s*[\"']([^\"']{8,})[\"']"
    ),
    re.compile(
        r"(?i)(?:[a-z0-9_-]*(?:api[_-]?key|secret|token|password|passwd))"
        r"\s*[:=]\s*([A-Za-z0-9_./+=-]{16,})"
    ),
    re.compile(r"\b(eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bnpm_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bpypi-AgEI[A-Za-z0-9_-]{30,}\b"),
    re.compile(r"\bsk_live_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
ALLOW_TOKENS = (
    "ornek",
    "example",
    "buraya-",
    "cok-gizli",
    "cok_gizli",
    "stale-",
    "runtime-",
    "remote-",
    "dummy",
    "fake",
    "test-",
    "<redacted",
)
SKIP_SUFFIXES = {".csv", ".parquet", ".png", ".jpg", ".jpeg", ".pdf", ".whl"}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    fingerprint: str


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]


def scan_text(text: str, path: str) -> list[Finding]:
    if Path(path).suffix.lower() in SKIP_SUFFIXES or path == ".env.example":
        return []
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for pattern in PATTERNS:
            for match in pattern.finditer(line):
                value = match.group(1) if match.lastindex else match.group(0)
                if any(token in value.lower() for token in ALLOW_TOKENS):
                    continue
                findings.append(Finding(path, number, _fingerprint(value)))
    return findings


def _git(args: list[str], root: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} basarisiz (exit={result.returncode})")
    return result.stdout


def scan_tracked(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for relative in _git(["ls-files"], root).splitlines():
        path = root / relative
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            findings.extend(scan_text(path.read_text(encoding="utf-8"), relative))
        except UnicodeDecodeError:
            continue
    return findings


def scan_history(root: Path) -> list[Finding]:
    """Tum commit patch'lerini tarar; bulguda sir degerini ASLA basmaz."""
    patch = _git(["log", "-p", "--all", "--no-ext-diff", "--unified=0"], root)
    findings: list[Finding] = []
    current = "<git-history>"
    line_number = 0
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            line_number = 0
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            line_number = int(match.group(1)) if match else 0
            continue
        if line.startswith("+") and not line.startswith("+++"):
            findings.extend(scan_text(line[1:], current))
            line_number += 1
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    findings = scan_tracked(args.root)
    if args.history:
        findings.extend(scan_history(args.root))
    unique = sorted(set(findings), key=lambda item: (item.path, item.line, item.fingerprint))
    for finding in unique:
        print(f"SECRET? {finding.path}:{finding.line} fingerprint={finding.fingerprint}")
    print(f"Secret scan: {len(unique)} bulgu")
    return 1 if unique else 0


if __name__ == "__main__":
    raise SystemExit(main())
