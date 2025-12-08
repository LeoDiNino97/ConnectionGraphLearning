# Structured Learning of Consistent Connection Graphs

<h5 align="center">
     
[![arXiv](https://img.shields.io/badge/Arxiv-2512.05657-b31b1b.svg?logo=arXiv)]([https://arxiv.org/abs/2512.05657](https://arxiv.org/pdf/2510.11245))
 <br>

</h5>

> [!TIP]
> Cellular sheaves are becoming increasingly influential in signal processing and machine learning thanks to their ability to encode local-to-global relationships over networks. However, this expressive power comes with substantial design and learning challenges: specifying a sheaf that satisfies structural desiderata—or inferring it directly from data—can quickly become complex. Building on classical ideas from graph signal processing and focusing on vector bundles, i.e., sheaves whose fibers lie on the orthogonal manifold, we introduce a learning framework that jointly infers both the graph topology and the sheaf geometry. The latter is represented by orthogonal transformations along edges. We assume these edge transformations admit a factorization through local bases at each node, yielding what is known as a flat bundle or consistent connection Laplacian. This structural assumption leads to several advantages: it tightly couples the sheaf Laplacian with the underlying graph spectrum, reduces the number of parameters, and provides a controllable and interpretable model for sheaf learning.

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
![PyManOpt](https://img.shields.io/badge/pymanopt-yellow?style=for-the-badge&logo=python&logoColor=white&link=[https://github.com/pymanopt/pymanopt])
