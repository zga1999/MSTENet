



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
| Model     | mAP   | Angle | Lr schd | Batch Size | Configs | Download |
|---        |---    |---    |---      |---         |---      |-     |
| MSTENet-T | 81.59 | le90  | 1x      | 1×8        | [config](https://github.com/zga1999/MSTENet/blob/main/projects/mstenet/dota_mstenet-t_fpn_oriented_1x.py) |[model](https://github.com/zga1999/MSTENet/blob/main/projects/mstenet/dota_mstenet-t_fpn_oriented_1x.py) \| [log](https://github.com/zga1999/MSTENet/blob/main/projects/mstenet/dota_mstenet-t_fpn_oriented_1x.py)   |
| MSTENet-S | 82.72 | le90  | 1x      | 2×8        | [config](https://github.com/zga1999/MSTENet/blob/main/projects/mstenet/dota_mstenet-s_fpn_oriented_1x.py) | [model](https://github.com/zga1999/MSTENet/releases/download/Weights_Logs/dota_MSTENet-S_fpn_oriented_1x.pth) \| [log](https://github.com/zga1999/MSTENet/releases/download/Weights_Logs/dota_MSTENet-S_20260222_233323.log)   |



### DIOR
| Model     | mAP   | Angle | Lr schd | Batch Size | Configs | Download |
|---        |---    |---    |---      |---         |---      |---   |
| MSTENet-S | 68.84 | le90  | 1x      | 2×8        | [config](https://github.com/zga1999/MSTENet/blob/main/projects/mstenet/dior_mstenet-s_fpn_oriented_1x.py) |[model](https://github.com/zga1999/MSTENet/blob/main/projects/mstenet/dota_mstenet-t_fpn_oriented_1x.py) \| [log](https://github.com/zga1999/MSTENet/blob/main/projects/mstenet/dota_mstenet-t_fpn_oriented_1x.py) |





### HRSC2016
| Model    | mAP(07) | mAP(12) | Angle | Lr schd | Batch Size | Configs | Download |
|---       |---      |---      |---    |---      |---         |---      |---   |
| MSTENet-S| 90.67   | 98.52   | le90  | 1x      | 2×8        | [config](https://github.com/zga1999/MSTENet/blob/main/projects/mstenet/hrsc2016_mstenet-s_fpn_oriented_1x.py) | [model](https://github.com/zga1999/MSTENet/blob/main/projects/mstenet/dota_mstenet-t_fpn_oriented_1x.py) \| [log](https://github.com/zga1999/MSTENet/blob/main/projects/mstenet/dota_mstenet-t_fpn_oriented_1x.py) | 

