This code is adapted from [HOC-Search](https://arxiv.org/abs/2309.06107) for generating views and instance segmentations 
for individual objects.
 
**[Dislaimer]** We don't maintain code in this subdirectory. 
 
---
### Setup

---
0) `cd` into `scannotate_preprocessing` directory, assuming the current directory is the root of this repository :

```bash
cd scannotate_preprocessing
```

1) We suggest creating a new environment:

```bash
conda env create -f environment.yml
conda activate hoc_preprocessing
```

2) Install [SAM](https://github.com/facebookresearch/segment-anything.git)

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

3) Download SAM weights:

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

### Extracting Views and Masks

---
0) Make sure you are in the `scannotate_preprocessing` directory.

1) Modify config script `config/0_Scannotate_masks.ini` and adapt paths.

`SCANNOTATE_PATH` should point to the base of [SCANnotate dataset](https://github.com/stefan-ainetter/SCANnotateDataset).
`out_folder` is the folder containing SCANnotate annotations and where instance masks will be extracted. Structure should
be as follows:

```
├── <SCANNOTATE_PATH>
│   ├── annotations
│   │    ├── <scene_name>
│   │    │    ├── <scene_name>.pkl
│   │    ├── ... 
```

`SCANNET_base_path` is the base path where [ScanNet](https://github.com/ScanNet/ScanNet) scenes are located with structure:
```
├── <SCANNET_base_path>
│   ├── <scene_name>
│   │    ├── color
│   │    ├── depth
│   │    ├── intrinsic
│   │    ├── pose
│   │    ├── <scene_name>.pkl
│   │    ├── ...
│   ├── ...
```

---
2) Render all 2D masks from 3D:
```bash
python ScanNet_renderer/Render_Scannotate_masks.py
```

Masks will be saved to `SCANNOTATE_PATH`:

```
├── <SCANNOTATE_PATH>
│   ├── annotations
│   │    ├── <scene_name>
│   │    │    ├── <scene_name>.pkl
│   │    │    ├── all_inst_seg_2d
│   │    │    ├── mask2d_from_3d
│   │    ├── ... 
```

---
3) Finally, calculate SAM masks for each object:
```bash
python ScanNet_renderer/Scannotate_2d_mask_predictor.py
```

Masks will be saved to `SCANNOTATE_PATH`:

```
├── <SCANNOTATE_PATH>
│   ├── annotations
│   │    ├── <scene_name>
│   │    │    ├── <scene_name>.pkl
│   │    │    ├── all_inst_seg_2d
│   │    │    ├── mask2d_from_3d
│   │    │    ├── sam_results_path
│   │    │    ├── skeleton_vis
│   │    │    ├── valid_maps
│   │    ├── ... 
```
