# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

import numpy as np

from pinn_jax.geometry import Geometry
from pinn_jax.geometry.timedomain import GeometryXTime

from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple


class BC(ABC):
    """Boundary condition.

    Args:
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.
        component: The output component satisfying this BC.
    """

    def __init__(self, geom: Union[Geometry, GeometryXTime], on_boundary: Callable, component: Union[int, Tuple[int, ...]]):
        self.geom = geom
        self.on_boundary = lambda x, on: np.array(
            [on_boundary(x[i], on[i]) for i in range(len(x))]
        )
        self.component = component

    def filter_idx(self, X):
        return self.on_boundary(X, self.geom.on_boundary(X))

    def filter(self, X):
        return X[self.filter_idx(X)]

    def collocation_points(self, X):
        return self.filter(X)

    @abstractmethod
    def error(self, points, fields):
        pass


class ConstantDirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = const."""

    def __init__(self, geom: Union[Geometry, GeometryXTime], value: Union[float, np.ndarray], on_boundary: Callable,
                 component: int = 0):
        super(ConstantDirichletBC, self).__init__(geom, on_boundary, component)
        self.value = value

    def error(self, _, fields: np.ndarray):
        return fields[:, self.component] - self.value


class FunctionDirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = f(x) on a boundary of the domain"""

    def __init__(self, geom: Union[Geometry, GeometryXTime], func: Callable, on_boundary: Callable, component=0):
        super(FunctionDirichletBC, self).__init__(geom, on_boundary, component)
        self.func = func

    def error(self, points, fields):
        return fields[:, self.component] - self.func(points)[:, self.component]


class PeriodicBC(BC):
    """periodic boundary condition: f(x) = f(x + k x)"""
    def __init__(self, geom: Union[Geometry, GeometryXTime], on_boundary: Callable, component=0):
        super(PeriodicBC, self).__init__(geom, on_boundary, component)

    def collocation_points(self, X):
        X1 = self.filter(X)  # subset of `X` on the boundary
        X2 = self.geom.periodic_point(X1, self.component)  # mirror'd versions of `X1`
        return X1, X2

    def error(self, points, fields):
        pass


class PointSetBC:
    """point-set condition: y(x) = f(x) for a specified set of points {x}"""
    def __init__(self, points, values, component: int = 0):
        self.points = np.array(points)
        self.values = np.array(values)
        self.component = component
    
    def collocation_points(self, _):
        return self.points
    
    def error(self, _, fields):
        return fields[:, self.component:self.component+1] - self.values


class IC(ABC):
    def __init__(self, geom: GeometryXTime, component: int = 0):
        self.geom = geom
        self.component = component

    @abstractmethod
    def error(self, points, fields):
        pass


class ConstantIC(IC):
    """initial condition y(x) = y_0 on the time-boundary of the domain"""
    def __init__(self, geom: GeometryXTime, value: Union[np.ndarray, float], component: int = 0):
        super().__init__(geom, component)
        self.value = value

    def error(self, _, fields):
        return fields[:, self.component] - self.value


class FunctionIC(IC):
    """initial condition: y(x) = f(x) on the time-boundary of the domain"""
    def __init__(self, geom: GeometryXTime, func: Callable, component: int = 0):
        super().__init__(geom, component)
        self.func = func

    def error(self, points, fields):
        values = self.func(points)
        return fields[:, self.component:self.component+1] - values
