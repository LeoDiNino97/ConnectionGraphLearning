# Structured Learning of Consistent Connection Graphs

<h5 align="center">
     
[![arXiv](https://img.shields.io/badge/Arxiv-2512.05657-b31b1b.svg?logo=arXiv)]([https://arxiv.org/abs/2512.05657](https://arxiv.org/pdf/2510.11245))
 <br>

</h5>

> [!TIP]
> Cellular sheaves are becoming increasingly influential in signal processing and machine learning thanks to their ability to encode local-to-global relationships over networks. However, this expressive power comes with substantial design and learning challenges: specifying a sheaf that satisfies structural desiderata—or inferring it directly from data—can quickly become complex. Building on classical ideas from graph signal processing and focusing on vector bundles, i.e., sheaves whose fibers lie on the orthogonal manifold, we introduce a learning framework that jointly infers both the graph topology and the sheaf geometry: the latter is represented by orthogonal transformations along edges. We assume these edge transformations admit a factorization through local bases at each node, yielding what is known as a flat bundle or consistent connection Laplacian. This structural assumption leads to several advantages: it tightly couples the sheaf Laplacian with the underlying graph spectrum, reduces the number of parameters, and provides a controllable and interpretable model for sheaf learning.

## Dependencies  
### Using `conda` package manager

It is highly recommended to create a Conda environment before installing dependencies.  
In a terminal, navigate to the root folder and run:

```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate <env_name>
```
You're ready to go! 🚀

## Simulations 
This section provides the necessary commands to run the simulations required for the experiments. The commands execute different training scripts with specific configurations. 

### Inference of random graphs
```bash
python scripts/random_graphs.py -m dimensions.seed='range(0, 840, 42)' dimensions.ratio=1.5,5,15 solvers.SCGL.alpha=0.0025 solvers.SCGL.beta=30 graph=ER,RBF,SBM solver=SCGL,SPD,SLGP
python scripts/random_graphs_readout.py 
```
## Citation

If you find this code useful for your research, please consider citing the following paper:

```
@misc{dinino2025learningstructureconnectiongraphs,
      title={Learning the Structure of Connection Graphs}, 
      author={Leonardo Di Nino and Gabriele D'Acunto and Sergio Barbarossa and Paolo Di Lorenzo},
      year={2025},
      eprint={2510.11245},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.11245}, 
}
```

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![PyManOpt](https://img.shields.io/badge/pymanopt-orange?style=for-the-badge)
![w&b](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white) 
![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white) 
![Hydra](https://img.shields.io/badge/Hydra-89CFF0?style=for-the-badge&logo=hyperland&logoColor=white)
