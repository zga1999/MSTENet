



## Results and Models

### Pre-trained Backbones

ImageNet pre-trained backbone (Model 1): **Download: TBD**

ImageNet pre-trained backbone (Model 2): **Download: TBD**

## Results and Models

### Pre-trained Backbones

The MSTENet backbones are pre-trained on ImageNet for 300 epochs.

- **MSTENet-T**: Download: Coming soon (`weights/mstenet/mstenet-t_backbone.pth`)
- **MSTENet-S**: Download: Coming soon (`weights/mstenet/mstenet-s_backbone.pth`)

### DOTA-v1.0

| Model | mAP | Angle | Lr schd | Batch Size | Configs | Logs |
|---    |---  |---    |---      |---         |---      |-     |
| MSTENet-T | 81.59 | le90 | 1x | 1×8 | [config](https://github.com/zga1999/MSTENet/blob/main/projects/mstenet/dota_mstenet-t_fpn_oriented_1x.py) |  |
| MSTENet-S | 82.72 | le90 | 1x | 2×8 | [config](https://github.com/zga1999/MSTENet/blob/main/projects/mstenet/dota_mstenet-s_fpn_oriented_1x.py) |  |

**Note:** DOTA-v1.0 results are reported using the multi-scale cropping strategy described in the paper.

### DIOR

| Model | mAP | Angle | Lr schd | Batch Size | Configs | Download | Note |
|---|---|---|---|---|---|---|---|
| O-RCNN | 64.30 | - | - | - | - | - | - |
| LSKNet-S | 65.90 | - | - | - | - | - | - |
| PKINet-S | 67.03 | - | - | - | - | - | - |
| **MSTENet-S** | **68.84** | le90 | 1x | 2×8 | [config](https://github.com/zga1999/MSTENet/blob/main/projects/mstenet/dior_mstenet-s_fpn_oriented_1x.py) | Coming soon | 39.7M / 209G |

### HRSC2016

| Model | mAP(07) | mAP(12) | Angle | Lr schd | Batch Size | Configs | Download | Note |
|---|---|---|---|---|---|---|---|---|
| O-RCNN | 90.50 | 97.60 | - | - | - | - | - | - |
| LSKNet-S | 90.65 | 98.46 | - | - | - | - | - | - |
| PKINet-S | 90.70 | 98.54 | - | - | - | - | - | - |
| **MSTENet-S** | **90.67** | **98.52** | le90 | 1x | 2×8 | [config](https://github.com/zga1999/MSTENet/blob/main/projects/mstenet/hrsc2016_mstenet-s_fpn_oriented_1x.py) | Coming soon | 39.7M / 209G |

### Training Details

MSTENet is implemented with PyTorch and MMRotate using Oriented R-CNN as the detection framework. All models are trained on eight NVIDIA RTX 4090 GPUs with ImageNet-pretrained backbones.

- **Optimizer:** AdamW
- **Initial learning rate:** 0.0002
- **Weight decay:** 0.05
- **Training schedule:** 1x (12 epochs)
- **DOTA-v1.0 input:** 1024 × 1024 patches with 200-pixel overlap
- **DIOR input:** 800 × 800
- **HRSC2016 input:** 800 × 800 (as reported in the paper)
- **Evaluation GPU:** NVIDIA L20

Batch Size is expressed as the number of images per GPU multiplied by the number of training GPUs. The reported results are taken from the MSTENet paper; checkpoint downloads will be provided when available.

