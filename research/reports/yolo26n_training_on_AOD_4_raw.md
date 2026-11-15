# YOLO26n — trening na AOD-4

> Uporządkowany zapis logu skopiowanego z konsoli Ultralytics. Wszystkie wartości, komunikaty oraz ich kolejność zachowano bez zmian; zmodyfikowano wyłącznie układ dokumentu.

## Odczyt wiersza treningowego

| Pole | Znaczenie |
|---|---|
| `Epoch` | numer bieżącej epoki i liczba zaplanowanych epok |
| `GPU_mem` | wykorzystanie pamięci GPU |
| `box_loss` | strata lokalizacji ramek |
| `cls_loss` | strata klasyfikacji |
| `l1_loss` | strata L1 |
| `Instances` | liczba instancji w batchu |
| `Size` | rozmiar wejścia |

## 1. Trening i walidacja okresowa — epoki 47–230

```text
     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.948      0.945      0.969      0.648
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     47/250      21.7G      1.562       1.28   0.002606         45        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.951      0.944      0.968      0.648
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     48/250      21.7G       1.56       1.27   0.002623         40        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.949      0.945      0.969      0.649
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     49/250      21.7G      1.549      1.293   0.002605         36        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.952      0.946      0.969      0.652
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     50/250      21.7G      1.551      1.279    0.00259         32        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.951      0.946      0.969      0.653
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     51/250      21.7G      1.553      1.275   0.002573         42        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.954      0.945      0.969      0.653
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     52/250      21.7G      1.537      1.276   0.002609         57        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.948      0.945      0.969      0.654
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     53/250      21.7G      1.543      1.257   0.002584         44        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.949      0.947       0.97      0.655
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     54/250      21.7G      1.549      1.271   0.002547         39        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369      0.951      0.946       0.97      0.655
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     55/250      21.7G      1.544      1.268   0.002557         81        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.8it/s 6.2s
                   all       4514       6369      0.951      0.949       0.97      0.655
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     56/250      21.7G      1.538      1.265   0.002578         37        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.951      0.948       0.97      0.657
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     57/250      21.7G      1.532       1.24   0.002564         51        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.3it/s 6.8s
                   all       4514       6369      0.951      0.947      0.971      0.657
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     58/250      21.7G      1.548      1.274   0.002607         50        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.3s
                   all       4514       6369      0.947       0.95      0.969      0.657
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     59/250      21.7G      1.535      1.235   0.002577         38        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.947       0.95      0.969      0.658
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     60/250      21.7G      1.547      1.241   0.002599         31        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.946       0.95      0.968      0.658
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     61/250      21.7G      1.529       1.24   0.002587         32        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.947      0.948      0.969      0.659
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     62/250      21.7G      1.537       1.23   0.002615         30        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.949      0.947      0.968      0.659
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     63/250      21.7G      1.529      1.226   0.002585         35        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.948      0.947      0.969      0.659
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
Uploaded best.pt ✅
     64/250      21.7G      1.529      1.221   0.002538         40        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369      0.949      0.948      0.969      0.659
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     65/250      21.7G      1.527      1.196   0.002538         40        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.949      0.949      0.969       0.66
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     66/250      21.7G      1.523      1.183   0.002512         64        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.949      0.949      0.969       0.66
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     67/250      21.7G      1.525      1.194   0.002492         50        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369       0.95      0.948      0.969      0.661
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     68/250      21.7G      1.516      1.221   0.002517         66        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 36.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369      0.949      0.949      0.969      0.661
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     69/250      21.7G      1.524      1.217   0.002487         31        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369       0.95      0.949      0.969       0.66
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     70/250      21.7G      1.507      1.198   0.002527         42        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369       0.95      0.949      0.969      0.662
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     71/250      21.7G       1.52      1.196   0.002533         25        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369       0.95       0.95      0.969      0.662
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     72/250      21.7G      1.516      1.202   0.002529         43        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369       0.95      0.949       0.97      0.662
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     73/250      21.7G      1.512      1.193   0.002538         31        960: 100% ━━━━━━━━━━━━ 247/247 7.0it/s 35.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369       0.95      0.949       0.97      0.662
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     74/250      21.7G      1.508      1.166   0.002526         31        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.3s
                   all       4514       6369      0.951      0.948       0.97      0.662
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     75/250      21.7G      1.514      1.187   0.002493         40        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369       0.95      0.949       0.97      0.662
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     76/250      21.7G      1.509      1.172   0.002498         41        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369       0.95      0.949       0.97      0.662
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     77/250      21.7G      1.509       1.16   0.002484         28        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369       0.95      0.949      0.971      0.663
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     78/250      21.7G        1.5      1.169   0.002497         43        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.4s
                   all       4514       6369      0.949       0.95       0.97      0.663
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     79/250      21.7G      1.495      1.167   0.002479         39        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369       0.95      0.948       0.97      0.663
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     80/250      21.7G      1.499      1.174   0.002476         23        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.951      0.949       0.97      0.663
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     81/250      21.7G      1.499       1.18   0.002476         34        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.949      0.949       0.97      0.663
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     82/250      21.7G      1.497      1.184   0.002486         50        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.8it/s 6.2s
                   all       4514       6369       0.95      0.948       0.97      0.664
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     83/250      21.7G      1.496      1.138   0.002461         31        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369       0.95      0.949      0.969      0.664
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     84/250      21.7G       1.49      1.153   0.002472         45        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.949      0.949      0.969      0.664
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
Uploaded best.pt ✅
     85/250      21.7G      1.481      1.173   0.002489         44        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.4s
                   all       4514       6369       0.95      0.949       0.97      0.664
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     86/250      21.7G      1.494      1.159   0.002471         36        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.3it/s 6.7s
                   all       4514       6369      0.951       0.95       0.97      0.664
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     87/250      21.7G      1.493      1.157   0.002429         35        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.951       0.95       0.97      0.664
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     88/250      21.7G       1.48      1.131   0.002475         26        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369       0.95       0.95       0.97      0.664
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     89/250      21.7G      1.494       1.12    0.00244         44        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369       0.95       0.95       0.97      0.665
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     90/250      21.7G      1.492      1.132     0.0024         38        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 37.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369       0.95       0.95      0.969      0.665
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     91/250      21.7G       1.49      1.143   0.002441         40        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.3s
                   all       4514       6369      0.951       0.95       0.97      0.666
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     92/250      21.7G      1.484      1.121   0.002414         48        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369       0.95      0.951       0.97      0.665
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     93/250      21.7G      1.475      1.143   0.002429         44        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369       0.95       0.95       0.97      0.666
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     94/250      21.7G      1.474      1.137   0.002419         34        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.951      0.951       0.97      0.666
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     95/250      21.7G      1.466      1.114   0.002434         51        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.3it/s 6.8s
                   all       4514       6369      0.951       0.95       0.97      0.666
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     96/250      21.7G      1.476      1.101   0.002429         40        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.952       0.95       0.97      0.667
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     97/250      21.7G      1.477      1.116   0.002437         30        960: 100% ━━━━━━━━━━━━ 247/247 6.6it/s 37.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.3s
                   all       4514       6369      0.953      0.949       0.97      0.666
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     98/250      21.7G      1.473      1.112   0.002403         31        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369      0.953      0.949       0.97      0.667
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     99/250      21.7G      1.476      1.088   0.002419         20        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.952      0.949       0.97      0.666
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    100/250      21.7G      1.477      1.095   0.002403         29        960: 100% ━━━━━━━━━━━━ 247/247 6.6it/s 37.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369      0.952      0.949       0.97      0.667
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    101/250      21.7G      1.468      1.102   0.002388         32        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.951      0.949       0.97      0.667
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    102/250      21.7G      1.471      1.104   0.002393         39        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.952      0.949       0.97      0.667
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    103/250      21.7G      1.467      1.117   0.002398         43        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.951      0.949       0.97      0.667
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    104/250      21.7G      1.467      1.106   0.002391         36        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369      0.952      0.949       0.97      0.667
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    105/250      21.9G      1.454       1.08   0.002392         33        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.952      0.949       0.97      0.668
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
Uploaded best.pt ✅
    106/250      21.9G       1.45       1.08   0.002367         45        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.3it/s 6.8s
                   all       4514       6369      0.952      0.949       0.97      0.667
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    107/250      21.9G      1.445      1.074   0.002371         37        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369      0.952      0.949       0.97      0.667
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    108/250      21.9G      1.449      1.071   0.002359         35        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.952      0.949       0.97      0.667
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    109/250      21.9G      1.455      1.078    0.00238         54        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369      0.952      0.948       0.97      0.668
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    110/250      21.9G      1.459      1.061   0.002382         40        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.952      0.948       0.97      0.668
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    111/250      21.9G      1.443      1.074   0.002376         49        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.952      0.948       0.97      0.668
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    112/250      21.9G      1.439      1.073   0.002373         28        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.952      0.948       0.97      0.668
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    113/250      21.9G       1.43      1.081   0.002318         32        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.952      0.948       0.97      0.669
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    114/250      21.9G      1.439       1.07    0.00233         42        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.8it/s 6.2s
                   all       4514       6369      0.953      0.948       0.97      0.669
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    115/250      21.9G      1.433       1.06   0.002344         43        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369      0.952      0.947       0.97      0.669
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    116/250      21.9G      1.442      1.064   0.002301         22        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369      0.952      0.947       0.97      0.669
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    117/250      21.9G      1.431      1.063   0.002299         33        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.952      0.947       0.97      0.669
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    118/250      21.9G      1.426      1.053   0.002316         30        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369      0.953      0.947       0.97      0.669
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    119/250      21.9G      1.435      1.062   0.002335         28        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.953      0.947       0.97      0.669
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    120/250      21.9G      1.427      1.055   0.002293         26        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.953      0.947       0.97      0.669
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    121/250      21.9G       1.42      1.057   0.002298         46        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.2it/s 6.9s
                   all       4514       6369      0.954      0.948       0.97      0.669
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    122/250      21.9G      1.422      1.039   0.002299         44        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.955      0.949       0.97      0.669
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    123/250      21.9G      1.433      1.038   0.002277         45        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.954      0.948       0.97      0.669
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    124/250      21.9G      1.416       1.02   0.002293         32        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.955      0.948       0.97       0.67
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    125/250      21.9G       1.42      1.049   0.002295         34        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.955      0.948       0.97       0.67
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    126/250      21.9G       1.41       1.05   0.002311         50        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369      0.956      0.948       0.97       0.67
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
Uploaded best.pt ✅
    127/250      21.9G      1.401       1.03   0.002264         32        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.956      0.948       0.97       0.67
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    128/250      21.9G      1.412      1.013    0.00226         39        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.956      0.948       0.97       0.67
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    129/250      21.9G      1.409      1.054   0.002293         40        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.956      0.948       0.97       0.67
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    130/250      21.9G      1.415      1.004   0.002271         34        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.3s
                   all       4514       6369      0.955      0.948       0.97       0.67
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    131/250      21.9G        1.4      1.004    0.00228         33        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.955      0.948      0.971       0.67
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    132/250      21.9G        1.4      1.009   0.002261         33        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.3s
                   all       4514       6369      0.955      0.949       0.97       0.67
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    133/250      21.9G      1.401     0.9994   0.002253         32        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369      0.955      0.949      0.971      0.671
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    134/250      21.9G      1.397     0.9882   0.002247         32        960: 100% ━━━━━━━━━━━━ 247/247 6.6it/s 37.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.956      0.949      0.971      0.671
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    135/250      21.9G       1.39      1.003    0.00228         38        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.956      0.949      0.971      0.671
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    136/250      21.9G      1.398     0.9982   0.002218         44        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 37.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.956      0.949       0.97      0.672
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    137/250      21.9G      1.398     0.9921   0.002171         42        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.956      0.949       0.97      0.673
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    138/250      21.9G      1.392      1.009   0.002255         30        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.3it/s 6.8s
                   all       4514       6369      0.957      0.949       0.97      0.673
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    139/250      21.9G       1.39      1.004   0.002215         60        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.957      0.949       0.97      0.673
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    140/250      21.9G      1.371     0.9735   0.002204         37        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.958      0.948       0.97      0.673
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    141/250      21.9G      1.379     0.9981   0.002196         48        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.958      0.948      0.971      0.673
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    142/250      21.9G      1.375      0.991   0.002203         43        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.957      0.949       0.97      0.673
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    143/250      21.9G      1.369     0.9797   0.002171         29        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369      0.959      0.947       0.97      0.674
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    144/250      21.9G      1.368       0.97   0.002201         43        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.957      0.948       0.97      0.674
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    145/250      21.9G      1.374     0.9724   0.002188         48        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.959      0.947       0.97      0.674
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    146/250      21.9G      1.362     0.9614   0.002194         43        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.958      0.948       0.97      0.674
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    147/250      21.9G       1.35     0.9533   0.002156         43        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.959      0.947       0.97      0.675
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
Uploaded best.pt ✅
    148/250      21.9G      1.353     0.9619   0.002192         78        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369       0.96      0.948       0.97      0.675
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    149/250      21.9G      1.364      0.956   0.002166         37        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 37.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369       0.96      0.947       0.97      0.675
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    150/250      21.9G      1.363     0.9721   0.002198         71        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.1it/s 7.1s
                   all       4514       6369       0.96      0.947      0.971      0.675
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    151/250      21.9G      1.349     0.9634   0.002166         46        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369       0.96      0.947      0.971      0.675
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    152/250      21.9G      1.358      1.001   0.002157         24        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.3it/s 6.8s
                   all       4514       6369       0.96      0.947      0.971      0.675
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    153/250      21.9G       1.35     0.9596   0.002159         39        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369       0.96      0.947      0.971      0.675
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    154/250      21.9G      1.351     0.9428   0.002137         53        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 37.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369       0.96      0.946      0.971      0.676
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    155/250      21.9G      1.347     0.9498   0.002146         49        960: 100% ━━━━━━━━━━━━ 247/247 6.6it/s 37.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369       0.96      0.946      0.971      0.676
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    156/250      21.9G      1.343     0.9379   0.002129         29        960: 100% ━━━━━━━━━━━━ 247/247 6.6it/s 37.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369       0.96      0.946      0.971      0.677
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    157/250      21.9G      1.343     0.9554   0.002121         35        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369       0.96      0.946      0.971      0.677
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    158/250      21.9G      1.349     0.9354   0.002119         49        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.959      0.946      0.971      0.677
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    159/250      21.9G      1.329     0.9307   0.002127         49        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 37.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369       0.96      0.946      0.971      0.677
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    160/250      21.9G      1.324     0.9321   0.002076         35        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.3s
                   all       4514       6369      0.959      0.946      0.971      0.677
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    161/250      21.9G      1.331     0.9138   0.002072         46        960: 100% ━━━━━━━━━━━━ 247/247 6.6it/s 37.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.2it/s 6.9s
                   all       4514       6369      0.958      0.947      0.971      0.677
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    162/250      21.9G      1.321     0.9142   0.002085         31        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 37.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.958      0.946      0.971      0.677
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    163/250      21.9G      1.322     0.8969    0.00207         36        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 37.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369      0.959      0.945      0.971      0.678
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    164/250      21.9G      1.322     0.9254    0.00208         56        960: 100% ━━━━━━━━━━━━ 247/247 6.6it/s 37.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.3s
                   all       4514       6369      0.959      0.946      0.971      0.678
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    165/250      21.9G      1.329     0.9079    0.00207         41        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.958      0.946      0.971      0.678
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    166/250      21.9G      1.305     0.9061    0.00208         40        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369      0.958      0.947      0.971      0.678
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    167/250      21.9G      1.307      0.904   0.002063         35        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.958      0.947      0.971      0.678
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    168/250      21.9G      1.309      0.896    0.00203         53        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369      0.958      0.948      0.971      0.679
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
Uploaded best.pt ✅
    169/250      21.9G      1.299     0.9148   0.002054         37        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.3it/s 6.7s
                   all       4514       6369      0.959      0.948      0.971      0.678
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    170/250      21.9G        1.3     0.8976    0.00208         29        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.3it/s 6.8s
                   all       4514       6369      0.959      0.947      0.971      0.679
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    171/250      21.9G      1.301     0.8822   0.002012         43        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369      0.959      0.948      0.971      0.679
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    172/250      21.9G       1.29      0.879   0.002088         33        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 37.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.959      0.948      0.971      0.679
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    173/250      21.9G      1.293     0.8838   0.002023         22        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 36.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369      0.959      0.947      0.971      0.679
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    174/250      21.9G      1.291     0.8709   0.002028         52        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369       0.96      0.947      0.971      0.679
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    175/250      21.9G      1.296     0.8887   0.002017         43        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.3s
                   all       4514       6369       0.96      0.948      0.971       0.68
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    176/250      21.9G      1.292     0.8752   0.002041         40        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369       0.96      0.948      0.972       0.68
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    177/250      21.9G      1.291     0.8697   0.002018         37        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 37.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.961      0.946      0.972       0.68
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    178/250      21.9G      1.274     0.8796   0.001977         35        960: 100% ━━━━━━━━━━━━ 247/247 6.6it/s 37.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.3it/s 6.8s
                   all       4514       6369      0.961      0.947      0.971       0.68
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    179/250      21.9G      1.277     0.8551    0.00196         52        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.961      0.947      0.971      0.681
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    180/250      21.9G      1.281     0.8642   0.001991         46        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369       0.96      0.947      0.971       0.68
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    181/250      21.9G      1.261     0.8601   0.001976         37        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.961      0.947      0.971      0.681
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    182/250      21.9G      1.265      0.872   0.001988         32        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369      0.961      0.947      0.971      0.681
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    183/250      21.9G      1.258     0.8553   0.001969         35        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 37.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.4s
                   all       4514       6369      0.961      0.947      0.971      0.681
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    184/250      21.9G      1.257     0.8436    0.00197         42        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.961      0.947      0.971      0.681
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    185/250      21.9G      1.261      0.868   0.001975         36        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.961      0.947      0.972      0.681
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    186/250      21.9G       1.26      0.841   0.001944         57        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.4s
                   all       4514       6369       0.96      0.949      0.972      0.681
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    187/250      21.9G      1.254     0.8419   0.001948         27        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.961      0.947      0.972      0.682
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    188/250      21.9G      1.261     0.8381   0.001947         33        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.961      0.948      0.972      0.682
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    189/250      21.9G      1.244     0.8367   0.001922         51        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.961      0.948      0.972      0.682
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
Uploaded best.pt ✅
    190/250      21.9G      1.246     0.8424   0.001925         27        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.961      0.948      0.972      0.682
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    191/250      21.9G       1.24     0.8142   0.001904         32        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.961      0.948      0.972      0.682
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    192/250      21.9G      1.235     0.8189   0.001922         29        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 37.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369      0.961      0.948      0.972      0.683
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    193/250      21.9G      1.241     0.8159   0.001909         24        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.962      0.949      0.972      0.684
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    194/250      21.9G      1.228     0.8183   0.001875         39        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369      0.961      0.949      0.972      0.683
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    195/250      21.9G      1.233     0.8141   0.001893         31        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.961      0.949      0.972      0.683
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    196/250      21.9G      1.244     0.8175   0.001888         35        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.961      0.949      0.972      0.684
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    197/250      21.9G      1.228     0.8116   0.001897         24        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.961      0.948      0.972      0.684
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    198/250      21.9G      1.224     0.7994   0.001849         50        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 37.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369      0.961      0.949      0.972      0.684
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    199/250      21.9G      1.218     0.7984    0.00189         27        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.961      0.949      0.972      0.684
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    200/250      21.9G      1.222     0.8007   0.001864         22        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.4s
                   all       4514       6369      0.961      0.949      0.972      0.685
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    201/250      21.9G      1.224     0.8065   0.001845         41        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.4s
                   all       4514       6369      0.961      0.949      0.972      0.685
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    202/250      21.9G      1.207     0.7768   0.001829         43        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369       0.96      0.949      0.972      0.685
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    203/250      21.9G       1.21     0.8056   0.001864         44        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369       0.96       0.95      0.972      0.685
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    204/250      21.9G      1.214     0.7951   0.001871         32        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.3it/s 6.8s
                   all       4514       6369       0.96      0.949      0.972      0.685
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    205/250      21.9G      1.217     0.8092   0.001844         36        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369       0.96      0.949      0.972      0.685
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    206/250      21.9G      1.205     0.7884   0.001869         29        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369       0.96      0.949      0.972      0.685
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    207/250      21.9G      1.198     0.7902   0.001832         34        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.3it/s 6.7s
                   all       4514       6369       0.96      0.949      0.972      0.685
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    208/250      21.9G      1.203     0.7872   0.001832         43        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.961      0.949      0.972      0.685
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    209/250      21.9G      1.196     0.7795   0.001813         37        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.6s
                   all       4514       6369       0.96      0.949      0.972      0.685
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    210/250      21.9G      1.196     0.7823   0.001846         48        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 36.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.3it/s 6.7s
                   all       4514       6369       0.96      0.949      0.972      0.686
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
Uploaded best.pt ✅
    211/250      21.9G      1.193     0.7815   0.001799         31        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369       0.96      0.949      0.972      0.686
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    212/250      21.9G       1.19      0.772   0.001819         43        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 37.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369      0.961      0.949      0.972      0.686
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    213/250      21.9G      1.197      0.781   0.001831         29        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.8it/s 6.2s
                   all       4514       6369       0.96      0.949      0.972      0.686
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    214/250      21.9G      1.188     0.7849    0.00184         42        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369       0.96      0.949      0.972      0.686
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    215/250      21.9G      1.176     0.7558   0.001811         44        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.3s
                   all       4514       6369       0.96      0.949      0.972      0.686
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    216/250      21.9G      1.184     0.7756   0.001823         34        960: 100% ━━━━━━━━━━━━ 247/247 6.6it/s 37.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369       0.96      0.949      0.972      0.686
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    217/250      21.9G      1.179     0.7724   0.001791         33        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369       0.96      0.949      0.972      0.686
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    218/250      21.9G      1.176     0.7623    0.00178         25        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.8it/s 6.2s
                   all       4514       6369      0.961      0.949      0.972      0.686
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    219/250      21.9G      1.171     0.7724   0.001785         39        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.962      0.949      0.973      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    220/250      21.9G      1.176     0.7675   0.001786         40        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.962      0.949      0.973      0.686
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    221/250      21.9G      1.176     0.7557   0.001788         29        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.962      0.948      0.972      0.686
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    222/250      21.9G      1.182     0.7646   0.001774         36        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.9it/s 6.1s
                   all       4514       6369      0.962      0.948      0.972      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    223/250      21.9G      1.176     0.7701   0.001766         31        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.963      0.948      0.973      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    224/250      21.9G      1.176     0.7626    0.00176         56        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.9it/s 6.1s
                   all       4514       6369      0.963      0.948      0.973      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    225/250      21.9G      1.164     0.7642   0.001792         48        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.8it/s 6.2s
                   all       4514       6369      0.963      0.948      0.973      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    226/250      21.9G      1.176     0.7621   0.001782         65        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369      0.963      0.948      0.973      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    227/250      21.9G      1.173     0.7642   0.001807         38        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.3s
                   all       4514       6369      0.963      0.948      0.973      0.688
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    228/250      21.9G      1.169     0.7665   0.001787         46        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.963      0.947      0.973      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    229/250      21.9G       1.16     0.7466   0.001764         40        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.964      0.947      0.973      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    230/250      21.9G      1.166     0.7602   0.001755         34        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.964      0.947      0.973      0.687
```

## 2. Zmiana augmentacji

```text
Closing dataloader mosaic
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
```

## 3. Przebieg treningu — epoki 231–250

```text
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    231/250      21.9G      1.068       0.46   0.001747         17        960: 100% ━━━━━━━━━━━━ 247/247 6.5it/s 38.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.964      0.948      0.973      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
Uploaded best.pt ✅
    232/250      21.9G      1.055     0.4502   0.001712         25        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.965      0.947      0.973      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    233/250      21.9G      1.055     0.4509   0.001721         17        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.4s
                   all       4514       6369      0.964      0.947      0.973      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    234/250      21.9G      1.047     0.4445     0.0017         16        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 36.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.964      0.947      0.973      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    235/250      21.9G      1.045      0.443   0.001705         28        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.963      0.947      0.973      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    236/250      21.9G      1.047     0.4406   0.001693         22        960: 100% ━━━━━━━━━━━━ 247/247 7.1it/s 34.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.3s
                   all       4514       6369      0.964      0.947      0.973      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    237/250      21.9G      1.037     0.4426   0.001676         40        960: 100% ━━━━━━━━━━━━ 247/247 7.0it/s 35.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.964      0.947      0.973      0.687
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    238/250      21.9G      1.037     0.4387   0.001675         18        960: 100% ━━━━━━━━━━━━ 247/247 7.0it/s 35.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369      0.964      0.947      0.973      0.688
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    239/250      21.9G      1.038     0.4379   0.001678         17        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 36.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.964      0.946      0.973      0.688
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    240/250      21.9G      1.035     0.4338   0.001664         24        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.9it/s 6.1s
                   all       4514       6369      0.964      0.946      0.973      0.688
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    241/250      21.9G      1.032     0.4364   0.001664         23        960: 100% ━━━━━━━━━━━━ 247/247 7.0it/s 35.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.7it/s 6.4s
                   all       4514       6369      0.964      0.947      0.973      0.688
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    242/250      21.9G      1.028      0.432   0.001648         36        960: 100% ━━━━━━━━━━━━ 247/247 7.1it/s 34.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.964      0.946      0.973      0.688
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    243/250      21.9G      1.028     0.4312   0.001645         18        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369      0.964      0.946      0.973      0.688
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    244/250      21.9G      1.029     0.4323    0.00165         18        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.6s
                   all       4514       6369      0.964      0.946      0.973      0.688
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    245/250      21.9G      1.024      0.432   0.001644         36        960: 100% ━━━━━━━━━━━━ 247/247 6.8it/s 36.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.8it/s 6.2s
                   all       4514       6369      0.965      0.946      0.973      0.688
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    246/250      21.9G       1.02     0.4304   0.001629         26        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.965      0.946      0.973      0.688
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    247/250      21.9G      1.028      0.434   0.001659         22        960: 100% ━━━━━━━━━━━━ 247/247 7.0it/s 35.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.4it/s 6.7s
                   all       4514       6369      0.965      0.946      0.973      0.688
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    248/250      21.9G      1.024     0.4352   0.001646         31        960: 100% ━━━━━━━━━━━━ 247/247 6.9it/s 35.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.4s
                   all       4514       6369      0.965      0.945      0.973      0.688
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    249/250      21.9G      1.024     0.4344   0.001647         21        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.6it/s 6.5s
                   all       4514       6369      0.965      0.946      0.973      0.688
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
    250/250      21.9G      1.025       0.43   0.001633         27        960: 100% ━━━━━━━━━━━━ 247/247 6.7it/s 36.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.5it/s 6.5s
                   all       4514       6369      0.965      0.946      0.973      0.688
```

## 4. Zakończenie treningu i walidacja `best.pt`

```text
250 epochs completed in 3.011 hours.
Optimizer stripped from /var/tmp/tmpfa9j_qvm/aod4s/weights/last.pt, 5.6MB
Optimizer stripped from /var/tmp/tmpfa9j_qvm/aod4s/weights/best.pt, 5.6MB
Validating /var/tmp/tmpfa9j_qvm/aod4s/weights/best.pt...
Ultralytics 8.4.191 🚀 Python-3.12.3 torch-2.14.1+cu132 CUDA:2 (NVIDIA RTX PRO 6000 Blackwell Server Edition, 97252MiB)
YOLO26n summary (fused): 121 layers, 2,390,464 parameters, 0 gradients, 5.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 4.2it/s 8.7s
                   all       4514       6369      0.965      0.945      0.973      0.688
              airplane       1119       1625      0.969      0.948      0.977      0.718
                  bird        638       1557      0.977      0.966      0.979       0.75
                 drone       1485       1602      0.953      0.908       0.96      0.634
            helicopter       1158       1585      0.962      0.961      0.974       0.65
Speed: 0.2ms preprocess, 0.7ms inference, 0.0ms loss, 0.3ms postprocess per image
Results saved to /var/tmp/tmpfa9j_qvm/aod4s
```
