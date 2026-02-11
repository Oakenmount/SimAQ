Limited-Angle Segmentation with Simulated Acquisitions
=================================

This repository contains code and utilities for limited-angle tomographic
segmentation, including training, inference, and a demo workflow.

Highlights
----------

- `Simaq` class: the primary interface for running inference and
  reconstruction workflows (see `simaq.py`).
- Training entry point: `train.py` wires up data, model, loss, and
  optimization for experiments.
- Usage example: the notebook `simaq_demo.ipynb` provides a runnable,
  end-to-end example of Simaq usage.

Repository Layout
-----------------

- `simaq.py`: implementation of the `Simaq` class.
- `train.py`: training script for model experiments.
- `inference.py`: inference utilities and helpers.
- `unet3d2d.py`: model definition.
- `datagenerator.py`, `augmentations.py`: data loading and augmentation.
- `loss.py`: loss functions.
- `simaq_demo.ipynb`: practical usage walkthrough.
- `configs/default.yaml`: default configuration.

Prerequisites
------------
Running simulations and or training requires the following:
- CUDA compatible GPU
- [TorchRadon](github.com/matteo-ronchetti/torch-radon)


Simaq Class
-----------

The `Simaq` class in `simaq.py` is the primary API surface. It
encapsulates model loading, preprocessing, and inference routines.

Training Script
---------------

The `train.py` script is used for training or finetuning the model. It connects
configuration, dataset construction, model instantiation, loss selection,
and optimizer setup.

Demo Notebook
-------------

The notebook `simaq_demo.ipynb` contains a usage example showing how to:

- Instantiate and configure `Simaq`.
- Load or generate data.
- Run inference and visualize outputs.

Please use the notebook as the reference workflow when starting a new
experiment.

Citing Simaq
------------

If you use Simaq in your work, please cite the Simaq paper:

```
@misc{egebjerg2025simaq,
      title={SimAQ: Mitigating Experimental Artifacts in Soft X-Ray Tomography using Simulated Acquisitions}, 
      author={Jacob Egebjerg and Daniel Wüstner},
      year={2025},
      eprint={2508.10821},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2508.10821}, 
}
```
