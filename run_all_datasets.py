"""
Automate dataset switching for the fraud detection pipeline.

For each configured dataset this script:
1. Replaces the project-level data/ directory with the extracted zip contents.
2. Normalizes archives that contain a single top-level folder so main.py can read
   data/transactions.csv and the auxiliary JSON files.
3. Runs main.py.
4. Moves output.txt to the dataset-specific upload file.
5. Validates that the output contains one non-empty transaction ID per line.
6. Prints a summary with fraud counts and generated file names.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
RAW_OUTPUT = PROJECT_ROOT / "output.txt"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    output_file: str
    zip_patterns: tuple[str, ...]


DATASETS = (
    DatasetSpec(
        name="truman",
        output_file="output_truman.txt",
        zip_patterns=(
            "*Truman*Show*validation*.zip",
            "*Truman*validation*.zip",
            "*Truman*Show*train*.zip",
            "*Truman*train*.zip",
        ),
    ),
    DatasetSpec(
        name="brave",
        output_file="output_brave.txt",
        zip_patterns=(
            "*Brave*New*World*validation*.zip",
            "*Brave*validation*.zip",
            "*Brave*New*World*train*.zip",
            "*Brave*train*.zip",
        ),
    ),
    DatasetSpec(
        name="deus",
        output_file="output_deus.txt",
        zip_patterns=(
            "*Deus*Ex*validation*.zip",
            "*Deus*validation*.zip",
            "*Deus*Ex*train*.zip",
            "*Deus*train*.zip",
        ),
    ),
)


def candidate_roots(extra_roots: list[Path]) -> list[Path]:
    roots = [
        *extra_roots,
        PROJECT_ROOT,
        PROJECT_ROOT.parent,
        Path.home() / "Desktop",
    ]

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved.exists() and resolved not in seen:
            unique_roots.append(resolved)
            seen.add(resolved)
    return unique_roots


def find_zip(spec: DatasetSpec, roots: list[Path]) -> Path:
    for pattern in spec.zip_patterns:
        matches: list[Path] = []
        for root in roots:
            matches.extend(path for path in root.glob(pattern) if path.is_file())

        if matches:
            return sorted(matches, key=lambda path: (len(str(path)), str(path)))[0]

    searched = ", ".join(str(root) for root in roots)
    patterns = ", ".join(spec.zip_patterns)
    raise FileNotFoundError(
        f"Could not find zip for {spec.name}. Searched {searched} using {patterns}"
    )


def reset_data_dir() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)


def safe_extract(zip_path: Path) -> None:
    data_root = DATA_DIR.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)

            if not member.filename or member_path.parts[0] == "__MACOSX":
                continue
            if member_path.name in {".DS_Store"} or member_path.name.startswith("._"):
                continue

            target = (DATA_DIR / member.filename).resolve()
            if not target.is_relative_to(data_root):
                raise ValueError(f"Blocked unsafe zip member path: {member.filename}")

            archive.extract(member, DATA_DIR)


def normalize_data_dir() -> None:
    if (DATA_DIR / "transactions.csv").is_file():
        return

    transaction_files = sorted(
        DATA_DIR.glob("**/transactions.csv"),
        key=lambda path: (len(path.parts), str(path)),
    )
    if not transaction_files:
        raise FileNotFoundError(
            f"No transactions.csv found after extracting into {DATA_DIR}"
        )

    source_dir = transaction_files[0].parent
    if source_dir == DATA_DIR:
        return

    for item in source_dir.iterdir():
        destination = DATA_DIR / item.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.move(str(item), str(destination))

    top_level = DATA_DIR / source_dir.relative_to(DATA_DIR).parts[0]
    if top_level.exists() and top_level.is_dir():
        shutil.rmtree(top_level)


def prepare_dataset(zip_path: Path) -> None:
    reset_data_dir()
    safe_extract(zip_path)
    normalize_data_dir()

    required = ["transactions.csv", "users.json", "locations.json", "sms.json", "mails.json"]
    missing = [name for name in required if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Dataset {zip_path} is missing required files after extraction: {missing}"
        )


def run_main(online: bool) -> str:
    RAW_OUTPUT.write_text("")

    env = os.environ.copy()
    if not online:
        env["FRAUD_PIPELINE_OFFLINE"] = "1"
        env["OPENROUTER_API_KEY"] = ""
        env["LANGFUSE_PUBLIC_KEY"] = ""
        env["LANGFUSE_SECRET_KEY"] = ""
        env["LANGFUSE_HOST"] = ""
        env["LANGFUSE_TRACING_ENABLED"] = "false"

    result = subprocess.run(
        [sys.executable, str(MAIN_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    combined_log = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        if not RAW_OUTPUT.exists():
            RAW_OUTPUT.write_text("")
        raise RuntimeError(
            f"main.py failed with exit code {result.returncode}\n{combined_log}"
        )

    if not RAW_OUTPUT.is_file():
        RAW_OUTPUT.write_text("")
        raise FileNotFoundError("main.py completed but did not generate output.txt")

    return combined_log


def validate_output(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"{path.name} is empty")

    invalid_lines = [
        idx
        for idx, line in enumerate(lines, start=1)
        if not line or line != line.strip() or any(char.isspace() for char in line)
    ]
    if invalid_lines:
        preview = ", ".join(str(idx) for idx in invalid_lines[:10])
        raise ValueError(
            f"{path.name} must contain exactly one ID per line; bad lines: {preview}"
        )

    return lines


def process_dataset(spec: DatasetSpec, roots: list[Path], online: bool) -> tuple[str, Path, int]:
    zip_path = find_zip(spec, roots)
    print(f"[{spec.name}] using {zip_path}")

    final_output = PROJECT_ROOT / spec.output_file
    # Safety: never overwrite an existing upload file. Require manual removal first.
    if final_output.exists():
        raise FileExistsError(
            f"Target output file already exists: {final_output}. To proceed, remove or move this file before running the batch."
        )

    prepare_dataset(zip_path)
    run_main(online=online)

    # Move the freshly-created output.txt into the dataset-specific name.
    if not RAW_OUTPUT.exists():
        raise FileNotFoundError("main.py completed but did not produce output.txt")

    RAW_OUTPUT.replace(final_output)

    fraud_ids = validate_output(final_output)
    print(f"[{spec.name}] generated {final_output.name} ({len(fraud_ids)} frauds)")
    return spec.name, final_output, len(fraud_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract each fraud dataset, run main.py, and generate upload files."
    )
    parser.add_argument(
        "--zip-root",
        action="append",
        default=[],
        type=Path,
        help="Additional directory to search for dataset zip files. Can be passed multiple times.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow live OpenRouter and Langfuse calls. By default batch generation runs offline for deterministic outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = candidate_roots(args.zip_root)

    summary: list[tuple[str, Path, int]] = []
    for spec in DATASETS:
        summary.append(process_dataset(spec, roots, online=args.online))

    print("\nSummary")
    print("-------")
    for name, output_path, fraud_count in summary:
        print(f"{name}: {fraud_count} frauds -> {output_path.name}")

    print("\nGenerated files:")
    for _, output_path, _ in summary:
        print(output_path.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
