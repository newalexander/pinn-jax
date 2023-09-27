# pinn-jax

This library implements physics-informed neural networks (PINNs) in the JAX framework.

### License and Copyright

Copyright 2023 Johns Hopkins University Applied Physics Laboratory

Licensed under the Apache License, Version 2.0

### Installation

`pip install -e pinn-jax/`

The key requirements are:
- `jax >= '0.3.23'`
- `flax >= '0.6.1'`
- `optax >= '0.1.3'`
- `chex >= '0.1.5'`
- `jaxtyping >= '0.2.7'`

### Use

The `burgers.py` example in `examples/` shows general use of how to use `pinn-jax` to solve the Burger's equation 
(a nonlinear, time-dependent PDE) using PINNs.

For further use, see documentation for each class and function
- Different PDEs are implemented in the `equations` module
  - Derivatives with respect to NN inputs are calculated using functions from the `derivatives.py` module
- Different benchmark problems are implemented in the `benchmarks` module
- Different PINN approaches are implemented in the `problems` module
  - Network training is performed by defining a `get_train_step` method for each problem-type
- Different domain geometries are defined in the `geometry` module
  - This is based on the `geometry` component used in the `deepxde` library (https://github.com/lululxvi/deepxde/tree/master) 

The `pinn-jax` framework is easily extendable to novel types of PINN and systems of differential equations. This can be
done by subclassing the `PartialDiffEq` or `OrdinaryDiffEq` classes, defined the `problems` module.

### Citations

If you use `pinn-jax` in your work, please cite:

```
@INPROCEEDINGS{10089728,
  author={New, Alexander and Eng, Benjamin and Timm, Andrea C. and Gearhart, Andrew S.},
  booktitle={2023 57th Annual Conference on Information Sciences and Systems (CISS)}, 
  title={Tunable complexity benchmarks for evaluating physics-informed neural networks on coupled ordinary differential equations}, 
  year={2023},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/CISS56502.2023.10089728}}
```