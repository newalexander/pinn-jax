# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

import os
import json

from jax import random
from flax import serialization

from pinn_jax.optimization import utils

from abc import ABC, abstractmethod
from typing import Callable, Dict, List


class AbstractProblem(ABC):
    def __init__(self,
                 model: Callable,
                 config: Dict,
                 metric_functions: Dict[str, Callable],
                 additional_keys: List[str],
                 additional_info: List[str]):

        # ensure we have everything we need for the model to work
        self.required_keys = ['n_interior', 'n_epochs', 'n_input', 'lr', 'optimizer', 'log_every', 'save_every',
                              'random_state', 'model_path', 'dtype']
        self.required_keys += additional_keys  # anything needed for subclasses (e.g., additional seeds)
        if 'optimizer' in config.keys() and config['optimizer'] in ['adahessian', 'diagsg']:
            self.required_keys += ['key_r']
        self.config = config
        self._validate_config()  # make sure that `config` has everything we need
        assert config['dtype'] in ["float32", "float64"], "invalid data type specified!!!"

        # initialize state and optimizer
        key, key_x = random.split(random.PRNGKey(config['random_state']))
        optimizer_options = config.get('optimizer_options', None)
        # TODO: this assigns everything not in optax a unique interface, but this could be improved
        if config['optimizer'] in utils.OPTAX_OPTIMIZERS:
            config['optimizer_interface'] = 'optax'
        else:
            config['optimizer_interface'] = config['optimizer']
        self.state = utils.get_init_state(key, key_x, config['n_input'], model, config['lr'], config['optimizer'],
                                          config['dtype'], optimizer_options)

        self.loss_records = {'epoch': [], 'loss': []}
        self.error_records = {'epoch': []}
        self.metric_functions = metric_functions

        self.paths = self._make_paths()

        self.metric_records = {'epoch': []}
        for metric in self.metric_functions.keys():
            self.metric_records[metric] = []

        # add in any additional keys that might be needed for a particular algorithm
        for key in additional_info:
            self.loss_records[key] = []
            # self.error_records[key] = []
            # self.metric_records[key] = []

    def fit(self):
        """wrapper for each class's specific `train` method that saves things at the end"""
        with open(os.path.join(self.paths['data'], 'config.json'), 'w') as f:
            json.dump(self.config, f)

        self._train()

        self.save_records()
        self.save_params()

    """methods that need to be filled in for your specific problem type"""

    @abstractmethod
    def _train(self):
        pass

    @abstractmethod
    def get_loss_components(self):
        pass

    @abstractmethod
    def get_residual_loss_func(self):
        pass

    def _validate_config(self):
        for required_key in self.required_keys:
            assert required_key in self.config.keys(), "missing '" + required_key + "' `config` key!!!"

    """methods about storing information"""

    def save_params(self, epoch=None):
        file_name = 'model.jax' if epoch is None else f"model_{epoch}.jax"
        byte_reps = serialization.to_bytes(self.state.params)
        with open(os.path.join(self.paths['checkpoint'], file_name), 'wb') as f:
            f.write(byte_reps)

    def log_loss_records(self, epoch, loss, **kwargs):
        """`loss_records` is a `dict` of lists with keys 'epoch', 'loss', and any additional identifying info"""
        self.loss_records['epoch'].append(epoch)
        self.loss_records['loss'].append(float(loss))
        for key, value in kwargs.items():
            self.loss_records[key].append(float(value))

    def log_error_records(self, epoch, **kwargs):
        """`error_records` is a `dict` of lists with keys `epoch`, each loss component's name, and any identifying
        additional info"""
        self.error_records['epoch'].append(epoch)

        loss_components = self.get_loss_components()

        for loss_name, loss_value in loss_components.items():
            self.error_records[loss_name].append(loss_value)
        for key, value in kwargs.items():
            self.error_records[key].append(value)

    def log_metric_records(self, epoch, **kwargs):
        """`metric_records` is a `dict` of lists with keys, `epoch`, any specified loss function"""
        self.metric_records['epoch'].append(epoch)
        for metric_name, metric_func in self.metric_functions.items():
            self.metric_records[metric_name].append(float(metric_func(self.state.params)))
        for key, value in kwargs.items():
            self.metric_records[key].append(value)

    def save_records(self):
        with open(os.path.join(self.paths['data'], 'loss_records.json'), 'w') as f:
            json.dump(self.loss_records, f)
        with open(os.path.join(self.paths['data'], 'error_records.json'), 'w') as f:
            json.dump(self.error_records, f)
        with open(os.path.join(self.paths['data'], 'metric_records.json'), 'w') as f:
            json.dump(self.metric_records, f)

    def _make_paths(self, exist_ok=True):
        checkpoint_dir = os.path.join(self.config['model_path'], 'checkpoints')
        data_dir = os.path.join(self.config['model_path'], 'data')
        # plot_dir = os.path.join(self.config['model_path'], 'plots')

        os.makedirs(checkpoint_dir, exist_ok=exist_ok)
        os.makedirs(data_dir, exist_ok=exist_ok)
        # os.makedirs(plot_dir, exist_ok=exist_ok)

        return {'checkpoint': checkpoint_dir, 'data': data_dir}#, 'plot': plot_dir}
