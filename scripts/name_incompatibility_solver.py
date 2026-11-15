"""Przenieś wybraną liczbę zdjęć JPG do innego katalogu."""

import argparse
import shutil
from pathlib import Path


def next_available_name(path, used_names):
    """Znajdź kolejną wolną nazwę, zachowując osiem cyfr dla nazw liczbowych."""
    if len(path.stem) == 8 and path.stem.isascii() and path.stem.isdigit():
        start = int(path.stem)
        for offset in range(1, 100_000_000):
            name = f"{(start + offset) % 100_000_000:08d}{path.suffix}"
            if name not in used_names:
                return name
        raise RuntimeError("Brak wolnych ośmiocyfrowych nazw plików.")

    index = 1
    while (name := f"{path.stem}_{index}{path.suffix}") in used_names:
        index += 1
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True, help="Katalog źródłowy")
    parser.add_argument("--output_dir", type=Path, required=True, help="Katalog docelowy")
    parser.add_argument("--count", type=int, required=True, help="Liczba zdjęć do przeniesienia")
    parser.add_argument("--dry_run", action="store_true", help="Pokaż plan bez przenoszenia")
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count musi być dodatnie")
    if not args.source_dir.is_dir():
        parser.error(f"Katalog źródłowy nie istnieje: {args.source_dir}")
    if args.source_dir.resolve() == args.output_dir.resolve():
        parser.error("Katalog źródłowy i docelowy muszą być różne")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error(f"Cel nie jest katalogiem: {args.output_dir}")

    files = sorted(
        (path for path in args.source_dir.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if len(files) < args.count:
        parser.error(f"Za mało plików JPG: dostępne {len(files)}, wymagane {args.count}")
    files = files[:args.count]

    # Zachowaj oryginalne nazwy pozostałych wybranych plików, jeśli to możliwe.
    used_names = {path.name for path in args.output_dir.iterdir()} if args.output_dir.exists() else set()
    reserved_names = used_names | {path.name for path in files}
    plan = []
    for source in files:
        name = source.name
        if name in used_names:
            name = next_available_name(source, reserved_names)
        used_names.add(name)
        reserved_names.add(name)
        plan.append((source, args.output_dir / name))

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for source, target in plan:
        print(f"{source} -> {target}")
        if not args.dry_run:
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Plik docelowy już istnieje: {target}")
            shutil.move(str(source), str(target))
    print(f"{'Zaplanowano' if args.dry_run else 'Przeniesiono'} {len(plan)} plików.")


if __name__ == "__main__":
    main()
