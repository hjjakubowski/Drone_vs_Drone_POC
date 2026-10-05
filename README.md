# INZ HUD

MVP aplikacji desktopowej HUD w Pythonie. Projekt obejmuje frontend PySide6, nagrania testowe, miejsce na aktywny model detekcji, minimalne granice kodu dla widzenia maszynowego oraz nawigacji proporcjonalnej, a takze lekki katalog badawczy.

## Struktura MVP

- `app/hud_desktop/` - aplikacja desktopowa PySide6 i ekran HUD.
- `inz_project/vision/` - miejsce na integracje YOLO/RT-DETR.
- `inz_project/navigation/` - miejsce na algorytm nawigacji proporcjonalnej.
- `inz_project/runtime/` - miejsce na spinanie watkow, detekcji, nawigacji i GUI.
- `inz_project/video/` - miejsce na obsluge nagran i zrodel ramek.
- `datasets/recordings/` - nagrania i probki danych.
- `models/production/` - aktywny model uzywany przez aplikacje.
- `research/` - minimalna przestrzen na notebooki i eksperymenty.
- `configs/` - konfiguracja UI, runtime, modelu i nawigacji.
- `scripts/` - pomocnicze uruchamianie aplikacji.

## Docelowy sposob uruchamiania

Wersja koncowa powinna byc pakowana jako aplikacja desktopowa uruchamiana bez IDE. Na etapie MVP aplikacja uruchamia ekran wyboru widokow, odtwarza nagrania z `datasets/recordings/` i pokazuje placeholder trajektorii na HUD.

## Uruchomienie

```bash
.venv/bin/python -m app.hud_desktop.main
```
