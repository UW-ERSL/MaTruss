# MaTruss

[Integrating Material Selection with Design Optimization via Neural Networks](https://arxiv.org/abs/2112.12566)

[Aaditya Chandrasekhar*](https://aadityacs.github.io/), [Saketh Sridhara*](https://sakethsridhara.github.io/), [Krishnan Suresh](https://directory.engr.wisc.edu/me/faculty/suresh_krishnan)  

[Engineering Representations and Simulation Lab](https://ersl.wisc.edu)  

University of Wisconsin-Madison 



## Abstract
A critical step in topology optimization (TO) is finding sensitivities. Manual derivation and implementation of the sensitivities can be quite laborious and error-prone, especially for non-trivial objectives, constraints and material models. An alternate approach is to utilize automatic differentiation (AD). While AD has been conceptualized over decades, and has also been applied in TO, wider adoption has largely been absent.

In this educational paper, we aim to reintroduce AD for TO and make it easily accessible through illustrative codes. In particular, we employ [JAX](https://github.com/google/jax)  , a high-performance Python library for automatically computing sensitivities from a user defined TO problem. The resulting framework, referred to here as AuTO, is illustrated through several examples in compliance minimization, compliant mechanism design and microstructural design.


*contributed equally
