# Structured Learning of Consistent Connection Graphs

<h5 align="center">
     
[![arXiv](https://img.shields.io/badge/Arxiv-2512.05657-b31b1b.svg?logo=arXiv)]([https://arxiv.org/abs/2512.05657](https://arxiv.org/pdf/2510.11245))
 <br>

</h5>

> [!TIP]
> Semantic communication systems aim to transmit task-relevant information between devices capable of artificial intelligence, but their performance can degrade when heterogeneous transmitter--receiver models produce misaligned latent representations. Existing semantic alignment methods typically rely on additional digital processing at the transmitter or receiver, increasing overall device complexity. In this work, we introduce the first over-the-air semantic alignment framework based on stacked intelligent metasurfaces (SIM), which enables latent-space alignment directly in the wave domain, reducing substantially the computational burden at the device level. We model SIMs as trainable linear operators capable of emulating both supervised linear aligners and zero-shot Parseval-frame-based equalizers. To realize these operators physically, we develop a gradient-based optimization procedure that tailors the metasurface transfer function to a desired semantic mapping. Experiments with heterogeneous vision transformer (ViT) encoders show that SIMs can accurately reproduce both supervised and zero-shot semantic equalizers, achieving up to 90% task accuracy in regimes with high signal-to-noise ratio (SNR), while maintaining strong robustness even at low SNR values.

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
