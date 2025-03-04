<div align="center">
	<img width = "33%" src="__meta__/logo.jpg">

<p align=center> <b> PyTorchGeoNodes is a differentiable module for reconstructing 3D objects from images using interpretable shape programs.
</b></p>

<a href="https://vevenom.github.io/pytorchgeonodes/">Project Page</a> |
<a href="https://arxiv.org/abs/2404.10620">Paper</a>
</div>



---
## Overview
**PyTorchGeoNodes** enables differentiable procedural graphs in PyTorch that reimplement functionalities of Geometry Nodes in Blender. 
More exactly, for different node types of Geometry Nodes, we implement corresponding node types with same 
functionalities using PyTorch, and PyTorch3D in case of geometric operations.

We provide algorithms for fitting parameters of PyTorchGeoNodes programs / procedural models, designed in Blender, to synthetic scenes
and scenes of the ScanNet dataset. In comparison to traditional CAD model retrieval methods, the use of shape programs for 3D reconstruction 
allows for reasoning about the semantic properties of reconstructed objects, editing, low memory footprint, etc.

---
## Updates 

- [ ] (Estimate February/March 2025)  Add integration of Gaussian Splatting into PyTorchGeoNodes. (The paper will be updated soon to explain the details)
- [x] (Estimate February/March 2025)  Add experiments for fitting objects from ScanNet scenes and data preparation scripts
- [x] January 2025 - Add genetic algorithm for fitting shape parameters to target 3D objects. (The paper will be updated soon to explain the details)
- [x] September 2024 - First release that includes a baseline combining coordinate descent and gradient descent for fitting shape parameters to synthetic scenes

---
### Setup

**Step (1)** From the root directory of this repository, create a new conda environment:

```bash
conda env create -f environment.yml
conda activate pytorchgeonodes
```

---
**Step (2)** Adjust paths config in `configs/general_config.yaml`:
```yaml
experiments_path_base: '<Base-Path-To-Experiments>'
processed_data_path: '<Folder-Where-Processed-Decision-Variables-Are-Saved>'
```

---
## Experiments

---
### `Synthetic Experiments:`  Simple gradient descent optimization

The following script demonstrates how to use the PytorchGeoNodes with the Adam optimizer to fit shape parameters of shape program, designed in Blender, to a synthetic scene:

```bash
python demo_optimize_pytorch_geometry_nodes.py --experiment_path demo_outputs/demo_optimize_pytorch_geometry_nodes
```

---

---
### `Synthetic Experiments:` Joint discrete and continuous optimization

**Step (1)**  Generate a synthetic dataset of scenes with chairs.

```bash
python generate_synthetic_dataset.py --category chair --num_scenes 10 --dataset_path synthetic_experiments/synthetic_dataset
```

---
**Step (2)** Preprocess shape parameters:

```bash

python preprocess_dv_values.py --category chair

```

---
**Step (3)**  Run the following command to reconstruct shape parameters of the chairs using genetic algorithm. Use ```--skip_refinement``` to run without refinement:

```bash
python reconstruct_synthetic_objects.py --category chair --dataset_name synthetic_dataset --experiment_path synthetic_experiments/ --method genetic
```

**[Note]** Current settings in `config/genetic_settings.yaml` were selected for accurate inference. By modifying parameters, including `population_size`, `num_offsprings`, `num_generations`, you effectively reduce computation time.  


Alternatively, you can change the `--method` to run reconstruction using coordinate descent baseline:

```bash
python reconstruct_synthetic_objects.py --category chair --dataset_path demo_outputs/demo_dataset --experiment_path demo_outputs/demo_dataset --method cd
```

---
**Step (4)**  Run evaluation script:

```bash
 python evaluate_synthetic_sp_parameters.py --category chair --synthetic_dataset_path synthetic_experiments/synthetic_dataset --experiments_path synthetic_experiments/ --experiment_name synthetic_dataset_genetic --solution_name 0best_0_solution.json 
 ```

---

---
### `ScanNet Experiments:` Joint discrete and continuous optimization

---
**Step (1)**  Prepare SCANnotate data. Follow the guide in [README.md](scannotate_preprocessing/README.md) for setting up ScanNet and SCANnotate, and for preprocessing data.

---
**Step (2)**  Adjust paths in `configs/scannotate_config.yaml`:
```yaml
scannet_processed_path: '<PATH-TO-SCANNET-SCANS>'
scannet_ply_path: '<PATH-TO-SCANNET-SCANS>'
scannotate_dataset_path: '<PATH-TO-SCANNOTATE-SCENES>'
scannotate_masks_path: '<PATH-TO-SCANNOTATE-MASKS-SCENES>'
```

---
**Step (3)**  If you have not done so, preprocess shape parameters for `OBJ_CAT` in `[chair, sofa, table]`:

```bash

python preprocess_dv_values.py --category <OBJ_CAT>

```

**Step (4)**  Run the following command to reconstruct shape parameters of the chairs using genetic algorithm. Use ```--skip_refinement``` to run without refinement:

```bash
python reconstruct_scannotate_objects.py --category OBJ_CAT --experiment_path scannotate_experiments/ --method genetic
```

**[Note]** Current settings in `config/genetic_settings.yaml` were selected for accurate inference. By modifying parameters, including `population_size`, `num_offsprings`, `num_generations`, you effectively reduce computation time.  

---
**Step (5)**  Run evaluation script on all categories

```bash 
python evaluate_scannotate_sp_parameters.py --experiments_path scannotate_experiments/ --experiment_name EXP_NAME --solution_name 0best_0_solution.json 
```

---
## Notes on Designing your Own Shape Programs

* When designing shape programs with Geometry Nodes feature of Blender, make sure that you use **Blender 4.0**. Structure of .blend files was changed with newer versions
of Blender and will likely lead to errors when compiling them to PyTorchGeoNodes code.

* When designing shape programs with Geometry Nodes feature of Blender, **check the supported nodes** first. Adding support for new nodes should
not be difficult as long as you understand the specific functionalities.

---
## Contributing

We introduced PyTorchGeoNodes with the goal of creating a framework for developing differentiable shape programs and simplifying their applications in 
tasks in 3D scene understanding. We are encouraging and welcoming contributions and integrations of new functionalities into PyTorchGeoNodes.

---
## Citation
If you find this code useful, please consider citing our paper:

```
@article{stekovic2024pytorchgeonodes,
  author    = {Stekovic, Sinisa and Ainetter, Stefan and D'Urso, Mattia and Fraundorfer, Friedrich},
  title     = {PyTorchGeoNodes: Enabling Differentiable Shape Programs for 3D Shape Reconstruction},
  journal   = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}
```
