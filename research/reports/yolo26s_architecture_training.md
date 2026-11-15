# Trening modelu YOLO26s

## Pobieranie zbioru danych

```text
Downloading https://storage.googleapis.com/alpha-ultralytics-eu/dataset-downloads/6af48ee95967a10144e97fdf/latest/drone-annotated.ndjson to '/tmp/tmpawj54oa9/hubert-jakubowski/datasets/drone-annotated/drone-annotated.ndjson': 100% ━━━━━━━━━━━━ 390.1KB 708.3KB/s 0.6s
```

## Architektura modelu

```text
Overriding model.yaml nc=80 with nc=4
                   from  n    params  module                                       arguments
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  2                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]
  3                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
  4                  -1  1    103360  ultralytics.nn.modules.block.C3k2            [128, 256, 1, False, 0.25]
  5                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]
  6                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5, 3, True]
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 13                  -1  1    477184  ultralytics.nn.modules.block.C3k2            [768, 256, 1, True]
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 16                  -1  1    136192  ultralytics.nn.modules.block.C3k2            [512, 128, 1, True]
 17                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 19                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]
 20                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 22                  -1  1   1843712  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True, 0.5, True]
 23        [16, 19, 22]  1    934960  ultralytics.nn.modules.head.Detect           [4, 1, True, [128, 256, 512]]
YOLO26s summary: 260 layers, 9,950,960 parameters, 9,950,960 gradients, 22.8 GFLOPs
```

## Przygotowanie treningu

```text
Transferred 696/708 items from pretrained weights
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 651.5±264.6 MB/s, size: 151.7 KB)
train: Scanning /datasets/drone-annotated-a5a6802e/labels/train.cache... 808 images, 62 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 808/808 109.3Mit/s 0.0s
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2214.4±1924.0 MB/s, size: 200.9 KB)
val: Scanning /datasets/drone-annotated-a5a6802e/labels/val.cache... 216 images, 25 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 216/216 75.5Mit/s 0.0s
optimizer: MuSGD(lr=0.005, momentum=0.937) with parameter groups 114 weight(decay=0.0), 126 weight(decay=0.001), 126 bias(decay=0.0)
Plotting labels to /var/tmp/tmpg5l4ac30/yolo26sdroneannotated/labels.jpg...
Using 808 train, 216 val images for fraction=1.0 at imgsz=960
Using 8 dataloader workers
Logging results to /var/tmp/tmpg5l4ac30/yolo26sdroneannotated
Starting training for 150 epochs...
```

## Przebieg treningu

```text
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      1/150      19.1G      2.006      95.03   0.002329         15        960: 100% ━━━━━━━━━━━━ 26/26 4.0it/s 6.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 2.9it/s 1.4s
                   all        216        191      0.798      0.684      0.778      0.376
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      2/150      19.1G      1.612      2.933   0.001841         22        960: 100% ━━━━━━━━━━━━ 26/26 7.5it/s 3.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 8.5it/s 0.5s
                   all        216        191          1       0.94      0.984      0.645
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      3/150      19.1G      1.559      1.809   0.001723         15        960: 100% ━━━━━━━━━━━━ 26/26 7.9it/s 3.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 8.8it/s 0.5s
                   all        216        191      0.968      0.969      0.982      0.568
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      4/150      19.1G       1.52      1.755   0.001668         14        960: 100% ━━━━━━━━━━━━ 26/26 8.3it/s 3.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 8.5it/s 0.5s
                   all        216        191      0.951      0.915      0.972      0.586
```
