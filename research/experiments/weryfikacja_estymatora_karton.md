# Wstępna weryfikacja estymatora na pomiarach kartonu

Porównano `ProportionalNavigation` z ręcznymi pomiarami z [pomiary.csv](pomiar_zblizania_dodatkowe/pomiary.csv). Karton o szerokości 0,38 m był nieruchomy, kamera poruszała się w jego kierunku. Przyjęto `fx=1509 px`, rozdzielczość 1920×1080 i czasy wypowiedzianych znaczników wskazane orientacyjnie przez autora: 3 m przy 0 s, 2,5 m przy 5 s, 2 m przy 7 s, 1,5 m przy 8,5 s i 1 m przy 10 s. `fx` wyznaczono wcześniej z pomiarów kartonu przy 1 i 2 m; nie jest wynikiem pełnej kalibracji kamery.

Do estymatora podano ręcznie zmierzone rozmiary jako syntetyczne stany z ID 1 i środkiem umownie ustawionym na środek obrazu, z pomocniczą kolejną numeracją próbek i `dt` równym odstępowi ich czasów. `vw` i `vh` ustawiono na zero; eksperyment nie weryfikuje TTC. Do porównania odległości w czasach znaczników zastosowano interpolację liniową oszacowanej głębokości pomiędzy próbkami. Średnią prędkość na przedziale 5–10 s wyznaczono z różnicy interpolowanych głębokości, nie ze średniej wygładzonych wyjść `closing_speed`.

| Czas | Znacznik | Z z estymatora | Różnica |
|---:|---:|---:|---:|
| 0 s | 3,00 m | 2,963 m | −0,037 m |
| 5 s | 2,50 m | 2,516 m | +0,016 m |
| 7 s | 2,00 m | 1,947 m | −0,053 m |
| 8,5 s | 1,50 m | 1,504 m | +0,004 m |
| 10 s | 1,00 m | 1,040 m | +0,040 m |

Średnia bezwzględna różnica dla czterech znaczników 5–10 s wynosi około 0,028 m. Średnia zmiana głębokości na tym przedziale to około 0,295 m/s wobec 0,300 m/s ze znaczników.

Jest to jedno nagranie i kilkanaście rzadkich próbek, a czasy znaczników nie są dokładnym pomiarem referencyjnym. Weryfikacja nie obejmuje działania YOLO, BoT-SORT, estymacji chwilowej w każdej klatce ani zmiennej orientacji drona E99. Wynik nie stanowi potwierdzenia dokładności fizycznych wskazań PN ani dowodu wykonanego manewru.
