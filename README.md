# MaTruss

[Integrating Material Selection with Design Optimization via Neural Networks](https://arxiv.org/abs/2112.12566)

[Aaditya Chandrasekhar*](https://aadityacs.github.io/), [Saketh Sridhara*](https://sakethsridhara.github.io/), [Krishnan Suresh](https://directory.engr.wisc.edu/me/faculty/suresh_krishnan)  

[Engineering Representations and Simulation Lab](https://ersl.wisc.edu)  

University of Wisconsin-Madison 



## Abstract
The engineering design process often entails optimizing the underlying geometry while simultaneously selecting a suitable material. For a certain class of simple problems, the two are separable where, for example, one can first select an optimal material, and then optimize the geometry. However, in general, the two are not separable. Furthermore, the discrete nature of material selection is not compatible with gradient-based geometry optimization, making simultaneous optimization challenging.

In this paper, we propose the use of variational autoencoders (VAE) for simultaneous optimization. First, a data-driven VAE is used to project the discrete material database onto a continuous and differentiable latent space. This is then coupled with a fully-connected neural network, embedded with a finite-element solver, to simultaneously optimize the material and geometry. The neural-network's built-in gradient optimizer and back-propagation are exploited during optimization.

The proposed framework is demonstrated using trusses, where an optimal material needs to be chosen from a database, while simultaneously optimizing the cross-sectional areas of the truss members. Several numerical examples illustrate the efficacy of the proposed framework

*contributed equally
