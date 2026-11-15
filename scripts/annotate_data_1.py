#!/usr/bin/env python3
import csv
from itertools import zip_longest
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets/recordings/data_1'
MODEL = ROOT / 'models/production/yolo26n_aod4_best.pt'
REPORT = DATA.parent / 'data_1_without_detections.csv'
SPLITS = {'train': 'train', 'test': 'test', 'val': 'valid'}
DRONE_LABEL_ID = 2
CONFIDENCE = 0.40
BATCH_SIZE = 16


def main():

    """Generate drone YOLO HBB pseudo-labels and list images without drone detections."""

    images = {name: sorted((DATA / folder / 'images').glob('*.jpg'))
              for name, folder in SPLITS.items()}
    for name, folder in SPLITS.items():
        if not images[name]:
            raise RuntimeError(f'No JPG images in {DATA / folder / "images"}')
        labels = DATA / folder / 'labels'
        if any((labels / f'{image.stem}.txt').exists() for image in images[name]):
            raise FileExistsError(f'Existing labels in {labels}; refusing to overwrite')
    if REPORT.exists():
        raise FileExistsError(f'Existing report: {REPORT}')

    model = YOLO(str(MODEL))
    names = model.names
    drone_indices = [int(index) for index, name in names.items() if name.lower() == 'drone']
    if len(drone_indices) != 1:
        raise ValueError(f'Unexpected model classes: {names}')
    drone_model_id = drone_indices[0]
    missing = {name: [] for name in SPLITS}
    for name, folder in SPLITS.items():
        labels = DATA / folder / 'labels'
        labels.mkdir(parents=True, exist_ok=True)
        for offset in range(0, len(images[name]), BATCH_SIZE):
            batch = images[name][offset:offset + BATCH_SIZE]
            results = model.predict([str(p) for p in batch], conf=CONFIDENCE,
                                    classes=[drone_model_id], batch=BATCH_SIZE,
                                    verbose=False, device='cpu')
            for image, result in zip(batch, results, strict=True):
                rows = []
                for box in result.boxes:
                    if (int(box.cls.item()) != drone_model_id
                            or float(box.conf.item()) <= CONFIDENCE):
                        continue
                    coords = box.xywhn[0].tolist()
                    rows.append(f'{DRONE_LABEL_ID} ' + ' '.join(f'{value:.6f}' for value in coords))
                if rows:
                    (labels / f'{image.stem}.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
                else:
                    missing[name].append(image.name)
            print(f'{name}: {min(offset + len(batch), len(images[name]))}/{len(images[name])}', flush=True)
    with REPORT.open('w', encoding='utf-8', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(SPLITS)
        writer.writerows(zip_longest(*(missing[name] for name in SPLITS), fillvalue=''))
    for name in SPLITS:
        print(f'{name}: {len(images[name]) - len(missing[name])} annotated, {len(missing[name])} without detections')
    print(f'Report: {REPORT}')


if __name__ == '__main__':
    main()
