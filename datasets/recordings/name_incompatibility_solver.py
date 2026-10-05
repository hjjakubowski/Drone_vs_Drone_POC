import argparse
import shutil
from pathlib import Path
from secrets import randbelow


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Move files from multiple source directories into one output directory "
            "while fixing duplicated filenames."
        )
    )
    parser.add_argument(
        "--source_dirs",
        nargs="+",
        required=True,
        help="Source directories containing files to merge.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination directory where all files will be moved.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show planned moves without changing files.",
    )

    return parser.parse_args()


def validate_source_dirs(source_dirs):
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Source directory does not exist: {source_dir}")


def generate_available_name(used_names, suffix):
    while True:
        candidate = f"{randbelow(100_000_000):08d}{suffix.lower()}"

        if candidate not in used_names:
            return candidate


def collect_files(source_dirs):
    files = []

    for source_dir in source_dirs:
        for path in source_dir.iterdir():
            if path.is_file():
                files.append(path)

    return files


def build_move_plan(files, output_dir):
    used_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    move_plan = []
    renamed_files = 0

    for source_path in files:
        target_name = source_path.name

        if target_name in used_names:
            target_name = generate_available_name(used_names, source_path.suffix)
            renamed_files += 1

        used_names.add(target_name)
        move_plan.append((source_path, output_dir / target_name))

    return move_plan, renamed_files


def move_files(move_plan, dry_run):
    moved_files = 0

    for source_path, target_path in move_plan:
        print(f"{source_path} -> {target_path}")

        if not dry_run:
            shutil.move(str(source_path), str(target_path))

        moved_files += 1

    return moved_files


def main():
    args = parse_args()

    source_dirs = [Path(source_dir) for source_dir in args.source_dirs]
    output_dir = Path(args.output_dir)

    validate_source_dirs(source_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(source_dirs)
    move_plan, renamed_files = build_move_plan(files, output_dir)
    moved_files = move_files(move_plan, args.dry_run)

    action = "Planned" if args.dry_run else "Moved"
    print(f"{action} {moved_files} files into: {output_dir}")
    print(f"Renamed {renamed_files} duplicated filenames.")


if __name__ == "__main__":
    main()
