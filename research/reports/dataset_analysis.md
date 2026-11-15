# Analiza i bezpieczne scalenie AOD-4 oraz ds_drone_hel_plane

Instrukcja dla studenta przygotowującego zbiór do detekcji obiektów latających w pracy inżynierskiej. Dotyczy detekcji HBB w Ultralytics YOLO26. Format etykiet jest standardowym formatem YOLO: `class_id x_center y_center width height`.

Dokument opisuje zadania, uzasadnienia, proponowane narzędzia i wymagane rezultaty. Implementację przygotowuje student; poniżej nie ma gotowego kodu potoku.

## 1. Cel, zakres i zasady

Celem jest zachowanie możliwie dużej liczby poprawnych, użytecznych obrazów przy usunięciu rzeczywistych powtórzeń i udokumentowaniu ograniczeń benchmarku. Sama liczba zachowanych zdjęć nie jest miarą jakości scalenia.

Obowiązują następujące ustalenia:

- Oryginalne obrazy i etykiety AOD-4 `valid/test` pozostają niezmienione bajtowo. Dopuszczalne są prefiksy nazw plików.
- Wszystkie zaakceptowane obrazy ds_drone_hel_plane, niezależnie od pierwotnego podziału, trafiają wyłącznie do treningu.
- Wspólne pochodzenie z filmu jest informacją o zależności danych. **Nie powoduje automatycznego wykluczenia całej sekwencji.**
- Duplikaty identyczne można usuwać automatycznie. Usunięcie obrazu podobnego wymaga bezpośredniego potwierdzenia wizualnego, wspomaganego metrykami.
- Zachowanie oryginalnego benchmarku przy pozostawieniu innych klatek tych samych filmów nie zapewnia separacji nagrań. Wyniki należy opisywać jako ocenę na oryginalnym podziale AOD-4, a nie jako pomiar generalizacji na całkowicie nowe nagrania.
- Źródła i wcześniej wygenerowany zbiór pozostają nienaruszone. Nowy wariant zapisuj osobno jako `datasets/AOD_4_merged_v2`. Jeśli katalog już istnieje, przerwij eksport i nadaj nową wersję; nie nadpisuj go.
- Nie zakładaj z góry liczby obrazów po filtracji. Każde odrzucenie musi mieć uzasadnienie zapisane w raporcie.

### Kontekst poprzedniej analizy

Poprzedni potok pozostawił 9 715 obrazów, w tym 2 960 treningowych. Odrzucał całe rozpoznane sekwencje powiązane z benchmarkiem i korzystał ze spójnych składowych grafu podobieństw. Nowy protokół zmienia oba mechanizmy podejmowania decyzji. Samo wyłączenie `PROTECT_KNOWN_VIDEO_GROUPS` w istniejącym notebooku nie wystarczy, jeśli nadal pozostanie przechodnie usuwanie obrazów na podstawie hashy percepcyjnych.

Poprzedni wynik jest punktem odniesienia do oceny skutków filtracji. Nie jest docelowym rozmiarem nowego zbioru.

## 2. Przygotowanie środowiska i rejestru pracy

Pracuj w środowisku projektu z Pythonem 3.11. Zapisz rzeczywistą ścieżkę interpretera, wersje bibliotek, ziarno losowania i parametry analizy. Korzystaj z lokalnego `requirements.txt`; jeśli wybierzesz dodatkową bibliotekę, odnotuj jej nazwę i wersję.

Ustal katalog główny projektu na podstawie rzeczywistego systemu plików. W tym środowisku jest nim `/home/hubert/ISA/INŻ`; skróconej ścieżki `/ISA/INŻ` nie należy wpisywać bez sprawdzenia, czy istnieje.

Prowadź trzy oddzielne rejestry:

1. Inwentarz próbek: źródło, pierwotny podział, ścieżki obrazu i etykiety, rozmiary, liczba instancji, stan walidacji oraz identyfikator próbki.
2. Rejestr problemów i decyzji: przyczyna, lokalizacja, ewentualny numer linii, sposób naprawy lub wykluczenia oraz wynik kontroli.
3. Rejestr podobieństw i pochodzenia: porównane obrazy, metryki, decyzja przeglądu oraz osobna informacja o nagraniu źródłowym i pewności jego rozpoznania.

**Proponowane narzędzia:** `pathlib`, `logging`, `tqdm`, `pandas.DataFrame`, `DataFrame.to_csv` i `importlib.metadata.version`.

**Rezultat etapu:** opis środowiska, parametrów i lokalizacji wyników oraz ustalony sposób identyfikacji każdej próbki.

## 3. Inwentaryzacja źródeł i potwierdzenie mapowania klas

Ponownie przeskanuj oba źródła:

- `datasets/AOD_4`;
- `datasets/ds_drone_hel_plane`.

W obecnym układzie AOD-4 podziały znajdują się bezpośrednio w katalogu zbioru: `{train,valid,test}/{images,labels}`. Wcześniejszy eksport miał katalog pośredni `Images`. Zawsze sprawdzaj stan aktualny. Nie wybieraj katalogu OBB ani adnotacji wielokątowych jako źródła etykiet HBB.

Odczytaj `data.yaml`, `dataset.yaml` lub `classes.txt`. Sprzeczne deklaracje wymagają wyjaśnienia przed remapowaniem. Historyczne ścieżki absolutne w YAML ds_drone_hel_plane nie wskazują obecnych katalogów i nie powinny bez sprawdzenia sterować odczytem plików.

Potwierdzone wcześniej mapowanie dołączanego zbioru jest następujące:

| ID źródłowe | Nazwa źródłowa | ID docelowe | Nazwa docelowa |
|---:|---|---:|---|
| 0 | AirPlane | 0 | airplane |
| 1 | Drone | 2 | drone |
| 2 | Helicopter | 3 | helicopter |

Docelowy standard obejmuje `0: airplane`, `1: bird`, `2: drone`, `3: helicopter`. Mapowanie ds_drone_hel_plane wynosi zatem `{0: 0, 1: 2, 2: 3}`. Nie przyjmuj go wyłącznie na podstawie nazw zdjęć lub wcześniejszego przykładu opisowego.

Dla każdego źródła i podziału policz obrazy, pliki etykiet `.txt`, pliki tła 0 B, brakujące pary, osierocone etykiety i instancje klas. Osobno raportuj obrazy i instancje: jeden obraz może zawierać wiele obiektów. Rozróżnij liczbę wszystkich plików tekstowych od liczby faktycznych etykiet wybranego eksportu.

Dotychczasowe liczby kontrolne:

| Źródło | train | valid | test | Łącznie |
|---|---:|---:|---:|---:|
| AOD-4 | 15 761 | 4 514 | 2 241 | 22 516 |
| ds_drone_hel_plane | 9 498 | 641 | 617 | 10 756 |
| Łącznie | 25 259 | 5 155 | 2 858 | 33 272 |

W obu źródłach stwierdzono łącznie 600 etykiet 0 B. Są to wyniki wcześniejszego skanu, które należy odtworzyć lub wyjaśnić ich zmianę.

**Proponowane narzędzia:** `Path.rglob`, `Path.stat`, `yaml.safe_load`, `collections.Counter`, `DataFrame.groupby`.

**Rezultat etapu:** inwentarz wejściowy, tabela rozkładu klas oraz zapis metadanych potwierdzających mapowanie.

## 4. Walidacja obrazów i etykiet HBB

Dla każdej niepustej linii wymagaj dokładnie pięciu pól: `class_id x_center y_center width height`. Sprawdź całkowite, dozwolone ID klasy, skończoność wszystkich liczb, zakres `[0,1]` dla współrzędnych oraz dodatnie `width` i `height`. Osobno sprawdź narożniki ramki wyliczone z jej środka i rozmiaru. Zapisz stosowaną tolerancję zaokrągleń, np. `1e-6` dla narożników; nie używaj jej do ukrywania dużych błędów geometrii.

W przypadku uszkodzonej adnotacji skieruj całą próbkę do naprawy lub kwarantanny. Nie usuwaj samej błędnej linii, pozostawiając widoczny obiekt bez etykiety. Nie konwertuj wielokątów i nie przycinaj ramek automatycznie bez udokumentowanej kontroli wyniku.

Zachowuj poprawne, jawnie puste etykiety. Brak pliku etykiety nie jest potwierdzonym tłem. Odróżnij też plik 0 B od pliku zawierającego wyłącznie białe znaki.

Dla obrazów sprawdź możliwość pełnego zdekodowania, rozmiary i orientację EXIF. Obrót obrazu wymaga odpowiedniego przekształcenia etykiet. Nie stosuj automatycznej korekty orientacji bez sprawdzenia zgodności geometrii. Naprawy wykonuj wyłącznie na kopiach i zapisuj ich pochodzenie.

W poprzedniej analizie znaleziono trzy pliki AOD-4 z wielokątami zamiast HBB, dwie ramki ds_drone_hel_plane wychodzące poza obraz i jeden obraz z nieprawidłową wartością orientacji EXIF. Zweryfikuj je ponownie. Błąd w `valid/test` blokuje bezwarunkowe zatwierdzenie eksportu: opisz go i uzgodnij rozwiązanie z opiekunem, bez cichej zmiany benchmarku.

**Proponowane narzędzia:** `PIL.Image.open`, `Image.verify`, `Image.load`, `Image.getexif`, `numpy.isfinite`, `logging`.

**Rezultat etapu:** lista poprawnych próbek i rejestr problemów z przyczyną, lokalizacją, decyzją oraz opisem naprawy, jeśli została wykonana.

## 5. Kontrola geometrii i kompletności adnotacji

Przy stałym ziarnie losowania przygotuj po 16 obrazów z każdego źródła, z ramkami i docelowymi nazwami klas. Przejrzyj je przed kopiowaniem. Dodatkowo obejrzyj największe ramki, najmniejsze obiekty oraz próbki wymagające naprawy.

Dla `drone=2` porównaj rozkłady znormalizowanych szerokości, wysokości, pola i proporcji ramek. Raportuj mediany i percentyle, w tym 95. percentyl. W poprzedniej analizie 95. percentyl szerokości wynosił około `0,17` w AOD-4 i `0,92` w ds_drone_hel_plane. To wskazówka do przeglądu, nie dowód błędnych adnotacji: duża ramka może wynikać zarówno z bliskiego obiektu, jak i nadmiaru tła.

Wyjaśnij znaczenie ciasności ramek dla estymacji odległości: w uproszczonym modelu stadiometrycznym odległość zależy odwrotnie od pozornego rozmiaru obiektu. Sama ramka HBB, BoT-SORT i filtr Kalmana nie zapewniają metrycznej odległości ani TTC bez dodatkowych założeń i kalibracji.

Nie zmieniaj rozmiarów ramek wyłącznie na podstawie rozkładów statystycznych.

**Proponowane narzędzia:** `matplotlib.pyplot.subplots`, `matplotlib.patches.Rectangle`, `seaborn.histplot`, `seaborn.scatterplot`, `DataFrame.quantile`.

**Rezultat etapu:** siatki kontrolne, wykresy i opis różnic jakościowych oraz skali obiektów.

## 6. Deduplikacja oparta na dowodach

### 6.1. Oddziel trzy rodzaje relacji

| Relacja | Postępowanie |
|---|---|
| Identyczne pliki albo identyczne zdekodowane piksele | Automatyczne rozpoznanie duplikatu i wybór reprezentanta według zasad benchmarku |
| Obrazy podobne, ale nieidentyczne | Wyszukanie kandydatów, metryki pomocnicze i ręczne potwierdzenie każdej pary uzasadniającej usunięcie |
| Wspólny film albo podobna nazwa pliku | Raport pochodzenia i zależności, bez automatycznego usuwania |

### 6.2. Oblicz hashe i wyszukaj kandydatów

Dla wszystkich obrazów oblicz SHA-256 pliku, hash zdekodowanych pikseli z uwzględnieniem wymiarów oraz 64-bitowe pHash i dHash. Zdefiniuj powtarzalny sposób dekodowania i reprezentacji kolorów. Transformacje stosowane do porównań nie mogą zmieniać źródeł. Jeśli pliku nie można zdekodować, zapisz ten fakt zamiast podstawiać sztuczny hash.

Dotychczasowe odległości pHash `4` i dHash `6` mogą być punktem startowym wyszukiwania kandydatów. Nie traktuj ich jako dowodu duplikatu. Ocenę progów wykonuj na ręcznie opisanych parach z danych treningowych, obejmujących także podobne niebo z różnymi małymi obiektami. Nie dobieraj progów na podstawie wyników modelu na `test_detection_and_tracking.py`.

Unikaj budowania pełnej macierzy porównań wszystkich obrazów. Możesz użyć indeksu BK-tree do wyszukiwania sąsiadów w odległości Hamminga.

### 6.3. Potwierdź każdą parę uzasadniającą usunięcie

Podczas przeglądu porównaj cały kadr oraz obiekt: jego położenie, sylwetkę, skalę i otoczenie. Identyczny widok nieba lub otoczenia nie wystarcza, jeśli obiekt zajmuje inne miejsce albo ma inne cechy.

SSIM może wspomagać ocenę po kontrolowanym dopasowaniu obrazów. Zapisz sposób zmiany rozmiaru, dopasowania i zakres wartości użyty w metryce. Wysoki globalny wynik dla jednolitego tła nie stanowi samodzielnej podstawy odrzucenia; obejrzyj również obszar obiektu. Wątpliwości związane z kadrowaniem, skalą i orientacją wymagają ręcznego rozstrzygnięcia.

Nie stosuj przechodniego usuwania podobieństw: relacje A≈B i B≈C nie wystarczają do usunięcia A jako duplikatu C. Każde odrzucenie obrazu podobnego musi wskazywać bezpośrednio porównany, zachowany obraz referencyjny. Grupowanie identycznych plików lub pikseli można traktować jako relację równoważności; nie przenoś tej zasady na podobieństwo percepcyjne.

Rejestruj status pary: potwierdzony duplikat, różne użyteczne obrazy albo przypadek nierozstrzygnięty. Brak decyzji nie uzasadnia usunięcia. Nierozstrzygnięte pary trening–benchmark trafiają do dalszej kontroli i blokują status końcowej gotowości zbioru.

**Proponowane narzędzia:** `hashlib.sha256`, `imagehash.phash`, `imagehash.dhash`, indeks BK-tree, `cv2.resize`, `cv2.absdiff`, opcjonalnie `skimage.metrics.structural_similarity` z biblioteki scikit-image.

**Rezultat etapu:** tabela porównanych par z metrykami, decyzją przeglądu, uzasadnieniem i identyfikatorem obrazu referencyjnego dla każdego odrzucenia.

## 7. Zachowanie benchmarku i wybór reprezentantów

Podejmuj decyzje w następującej kolejności:

1. Przy potwierdzonym duplikacie pomiędzy treningiem a AOD-4 `valid/test` zachowaj benchmark, a próbkę treningową wyklucz.
2. Przy powtórzeniach wewnątrz treningu zachowaj jednego reprezentanta. Domyślnie preferuj AOD-4, ale konflikty lub różnice kompletności etykiet najpierw sprawdź ręcznie. Nie łącz automatycznie sprzecznych adnotacji.
3. Powtórzenia i wspólne filmy pomiędzy AOD-4 `valid/test` raportuj bez zmieniania tych podzbiorów.
4. Wszystkie zaakceptowane próbki ds_drone_hel_plane ze źródłowych `train`, `valid` i `test_detection_and_tracking.py` skieruj do docelowego treningu.

Przy większej liczbie kopii zapisz dla każdej usuniętej próbki relację do obrazu, który rzeczywiście pozostaje w wyniku. Nie opieraj decyzji wyłącznie na podobieństwie do innego odrzuconego obrazu.

Prowadź osobny raport wspólnych nagrań. Rozróżniaj potwierdzone pochodzenie od przypuszczeń wynikających z nazw. Nie łącz grup filmowych z grafem podobieństw w celu automatycznego wykluczania całych składowych.

Nawet po zakończeniu tego etapu opisuj ochronę precyzyjnie: potwierdzone powtórzenia zostały usunięte zgodnie z zastosowanym protokołem, ale pełna niezależność nagrań nie jest zapewniona. Nie deklaruj uniwersalnego braku wycieku na podstawie samych hashy.

**Rezultat etapu:** ostateczny plan podziału, wybrani reprezentanci i osobny raport zależności pomiędzy źródłami oraz podzbiorami.

## 8. Opcjonalna analiza brakujących ptaków

Przewidź wariant z detektorem wytrenowanym wyłącznie na treningu AOD-4, bez obrazów benchmarku, albo modelem pretrenowanym na COCO. ID ptaka ustalaj przez `model.names`: w zbiorze docelowym wynosi `1`, a w indeksacji Ultralytics COCO `14`.

Dla wykryć z `conf > 0.45` wykonaj ręczną weryfikację przed dodaniem etykiety lub wykluczeniem obrazu. Sprawdzaj nakładanie z istniejącymi adnotacjami, aby uniknąć sprzecznych oznaczeń ptaka i drona. Dodanie ptaka nie może usuwać poprawnych etykiet innych obiektów.

Brak detekcji nauczyciela nie potwierdza braku nieoznaczonych ptaków. Wysoka pewność predykcji również nie zastępuje kontroli pseudoetykiety.

**Proponowane narzędzia:** `ultralytics.YOLO`, `YOLO.predict`, `model.names`, `Results.boxes.xywhn`, `Results.boxes.conf`.

**Rezultat etapu:** identyfikacja i hash modelu, jego wersja, próg, wykrycia oraz decyzje. Wyłączenie etapu trzeba jawnie wymienić w ograniczeniach zbioru. Nie pobieraj ani nie trenuj nauczyciela jako nieudokumentowanej części scalenia.

## 9. Plan eksportu i bezpieczne kopiowanie

Przed kopiowaniem przygotuj manifest zawierający identyfikator próbki, źródło, oryginalny i docelowy podział, ścieżki obrazu i etykiety, nazwy wynikowe, hashe oraz decyzje dotyczące adnotacji.

Nadaj prefiksy `aod4_` i `ds3c_`, uwzględnij pierwotny podział i stabilny identyfikator zapobiegający kolizjom. Sam prefiks źródła nie chroni przed powtórzeniem tej samej nazwy w różnych podziałach dołączanego zbioru. Sprawdź parowanie obraz–etykieta z rozróżnianiem wielkości liter; dodatkowo wykryj kolizje po jej ujednoliceniu.

Wynik ma zawierać `train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels` i `dataset.yaml`. W YAML zapisz rzeczywistą bezwzględną ścieżkę katalogu docelowego, `train: train/images`, `val: valid/images`, `test: test/images` i słownik czterech klas docelowych.

Wynik buduj w nowym katalogu tymczasowym na tym samym systemie plików. Kopiuj pliki, zamiast tworzyć dowiązania do źródeł. Przed publikacją przeprowadź pełny audyt kopii. Błąd może usuwać wyłącznie katalog tymczasowy danego przebiegu, nigdy źródła ani wcześniejszy wynik. Ponowne uruchomienie powinno rozpoznać zgodny manifest i zweryfikować wynik albo bezpiecznie odmówić nadpisania niezgodnej wersji.

**Proponowane narzędzia:** `shutil.copy2`, `tempfile.TemporaryDirectory`, `yaml.safe_dump`, `Path.rename`, `tqdm`.

Na końcu usuń pliki `*.cache` wyłącznie z wyniku i użyj `ultralytics.data.utils.check_det_dataset` z `autodownload=False`. Funkcja sprawdza konfigurację i ścieżki; nie zastępuje pełnego audytu obrazów, etykiet, parowania ani zależności danych.

**Rezultat etapu:** oddzielny, zweryfikowany katalog wynikowy, YAML, manifest i raport gotowości.

## 10. Raport do pracy inżynierskiej

Przygotuj tabele obrazów, instancji i próbek tła przed oraz po każdym etapie. Podaj liczby odrzuceń według przyczyn, osobno dla źródeł i podziałów. Nie sumuj nakładających się kategorii problemów jako liczby różnych usuniętych obrazów.

Pokaż rozkłady klas osobno dla treningu i benchmarku. Wyróżnij wkład ds_drone_hel_plane w docelowy trening. Porównaj liczebność i skład klas z poprzednim rygorystycznym wariantem, wyjaśniając wpływ zmiany reguł. Zwróć uwagę na ewentualną utratę reprezentacji poszczególnych klas i skal obiektów.

Dołącz siatki, rozkłady geometrii, manifest pochodzenia, rejestr napraw, decyzje deduplikacyjne i raport wspólnych filmów. Ryciny eksportuj do PNG w rozdzielczości 300 dpi oraz PDF; tabele do CSV i formatu wygodnego do wklejenia do pracy, np. HTML lub LaTeX.

Wyraźnie oddziel ustalenia wynikające z analizy od ograniczeń: niepełne rozpoznanie nagrań, możliwe pominięcia podobnych obrazów, niepewność etykiet i brakujących ptaków oraz zależności oryginalnych `valid/test`.

**Proponowane narzędzia:** `DataFrame.groupby`, `DataFrame.to_csv`, `DataFrame.to_html`, `DataFrame.to_latex`, `seaborn.barplot`, `Figure.savefig`.

## 11. Kryteria odbioru i przypadki kontrolne

Zbiór otrzymuje status gotowości dopiero po potwierdzeniu wszystkich warunków:

- Wszystkie oryginalne obrazy i etykiety AOD-4 `valid/test` są obecne, a ich hashe odpowiadają źródłom.
- Każdy eksportowany obraz ma właściwą etykietę o zgodnej nazwie; nie ma kolizji ani nieplanowanych plików.
- Każda etykieta spełnia wymagania HBB, a zaakceptowane etykiety tła są zachowane.
- Wszystkie próbki ds_drone_hel_plane znajdują się wyłącznie w treningu.
- Nie ma duplikatów identycznych pomiędzy treningiem a benchmarkiem według kontroli plików i pikseli.
- Każde usunięcie obrazu podobnego ma bezpośrednie, zapisane potwierdzenie i zachowany obraz referencyjny.
- Wszystkie wygenerowane pary kandydatów trening–benchmark są rozstrzygnięte; przypadków oczekujących nie przedstawia się jako zakończonego audytu.
- Wspólne filmy i powiązania oryginalnych `valid/test` są zaraportowane bez deklarowania pełnej niezależności nagrań.
- Kontrola konfiguracji Ultralytics kończy się powodzeniem, a ponowne wykonanie nie nadpisuje źródeł ani niezgodnego wyniku.

Sprawdź implementację na następujących przypadkach:

| Przypadek | Oczekiwane zachowanie |
|---|---|
| Identyczne obrazy o różnych nazwach | Rozpoznanie powtórzenia i zachowanie reprezentanta zgodnie z priorytetem benchmarku |
| Różne klatki tego samego filmu | Brak automatycznego usunięcia wyłącznie z powodu pochodzenia |
| Podobne niebo z różnymi obiektami | Przegląd pary; podobieństwo tła nie przesądza odrzucenia |
| Łańcuch A≈B≈C bez potwierdzenia A≈C | Brak przechodniego usunięcia A jako duplikatu C |
| Identyczny obraz z różnymi etykietami | Ręczne rozstrzygnięcie konfliktu, bez automatycznego łączenia adnotacji |
| Pusta etykieta 0 B | Zachowanie poprawnej próbki tła, jeśli nie wyklucza jej inna udokumentowana przyczyna |
| Brak etykiety albo niezgodność wielkości liter | Zgłoszenie problemu parowania, bez automatycznej zamiany na tło |
| Wielokąt, NaN/Inf, niedozwolona klasa lub niepoprawna geometria | Naprawa z kontrolą albo kwarantanna całej próbki |
| Nieprawidłowy EXIF lub uszkodzony obraz | Jawna decyzja jakościowa, bez niekontrolowanej zmiany geometrii |
| Duplikat pomiędzy treningiem a benchmarkiem | Zachowanie benchmarku i wykluczenie kopii treningowej |
| Powiązanie pomiędzy valid i test | Zachowanie obu podzbiorów i opis ograniczenia |
| Pseudoetykieta ptaka nakładająca się na drona | Ręczna weryfikacja konfliktu przed zmianą etykiet |

Do tych kontroli można wykorzystać mały zestaw syntetyczny, `pytest` oraz końcowe sprawdzenia inwariantów na rzeczywistym wyniku. Nie ma potrzeby uruchamiania pełnego treningu modelu, aby potwierdzić poprawność strukturalną scalenia. Poprawność strukturalna nie jest oceną jakości detektora.

## 12. Dokumentacja techniczna

- [Ultralytics: format zbiorów do detekcji](https://docs.ultralytics.com/datasets/detect/) — układ danych i pięciokolumnowe etykiety HBB.
- [Ultralytics: check_det_dataset](https://docs.ultralytics.com/reference/data/utils/#ultralytics.data.utils.check_det_dataset) — sprawdzanie konfiguracji zbioru i ścieżek.
- [Ultralytics: inferencja i obiekty Results](https://docs.ultralytics.com/modes/predict/) — opcjonalna analiza brakujących ptaków.
- [Ultralytics: klasy COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) — indeks klasy bird w modelach COCO.
- [ImageHash](https://github.com/JohannesBuchner/imagehash) — pHash, dHash i porównywanie hashy percepcyjnych.
- [scikit-image: structural_similarity](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity) — pomocnicza metryka SSIM.
