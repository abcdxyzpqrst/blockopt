# AdaBlock: SGD with Practical Block Diagonal Matrix Adaptation for Deep Learning (AISTATS 2022)
This repository contains the official implementations of the paper "AdaBlock: SGD with Practical Block Diagonal Matrix Adaptation for Deep Learning" publish in AISTATS 2022.
+ Jihun Yun (KAIST), Aurélie C. Lozano (IBM T.J. Watson Research Center), Eunho Yang (KAIST, AITRICS)

# Abstract
We introduce AdaBlock, a class of adaptive gradient methods that extends popular approaches such as Adam by adopting the simple and natural idea of using block-diagonal matrix adaption to effectively utilize structural characteristics of deep learning architectures. Unlike other quadratic or block-diagonal approaches, AdaBlock has complete freedom to select block-diagonal groups, providing a wider trade-off applicable even to extremely high-dimensional problems. We provide convergence and generalization error bounds for AdaBlock, and study both theoretically and empirically the impact of the block size on the bounds and advantages over usual diagonal approaches. In addition, we propose a randomized layer-wise variant of Adablock to further reduce computations and memory footprint, and devise an efficient spectrum-clipping scheme for AdaBlock to benefit from Sgd’s superior generalization performance. Extensive experiments on several deep learning tasks demonstrate the benefits of block diagonal adaptation compared to adaptive diagonal methods, vanilla Sgd, as well as modified versions of full-matrix adaptation.

# Acknowledgement
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2019-0-00075, Artificial Intelligence Graduate School Program(KAIST))
