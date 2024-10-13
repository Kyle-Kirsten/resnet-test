from copy import deepcopy
from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import norm
from ast import literal_eval as _eval

class SchedulerBase(ABC):
    def __init__(self):
        """Abstract base class for all Scheduler objects."""
        self.hyperparameters = {}

    def __call__(self, step=None, cur_loss=None):
        return self.learning_rate(step=step, cur_loss=cur_loss)

    def copy(self):
        """Return a copy of the current object."""
        return deepcopy(self)

    def set_params(self, hparam_dict):
        """Set the scheduler hyperparameters from a dictionary."""
        if hparam_dict is not None:
            for k, v in hparam_dict.items():
                if k in self.hyperparameters:
                    self.hyperparameters[k] = v

    @abstractmethod
    def learning_rate(self, step=None):
        raise NotImplementedError

class ConstantScheduler(SchedulerBase):
    def __init__(self, lr=0.01, **kwargs):
        """
        Returns a fixed learning rate, regardless of the current step.

        Parameters
        ----------
        initial_lr : float
            The learning rate. Default is 0.01
        """
        super().__init__()
        self.lr = lr
        self.hyperparameters = {"id": "ConstantScheduler", "lr": self.lr}

    def __str__(self):
        return "ConstantScheduler(lr={})".format(self.lr)

    def learning_rate(self, **kwargs):
        """
        Return the current learning rate.

        Returns
        -------
        lr : float
            The learning rate
        """
        return self.lr

    
class OptimizerBase(ABC):
    def __init__(self, lr, scheduler=None):

        self.cache = {}
        self.cur_step = 0
        self.hyperparameters = {}
        self.lr_scheduler = ConstantScheduler(lr)()

    def __call__(self, param, param_grad, param_name, cur_loss=None):
        return self.update(param, param_grad, param_name, cur_loss)

    def step(self):
        """Increment the optimizer step counter by 1"""
        self.cur_step += 1

    def reset_step(self):
        """Reset the step counter to zero"""
        self.cur_step = 0

    def copy(self):
        """Return a copy of the optimizer object"""
        return deepcopy(self)


    @abstractmethod
    def update(self, param, param_grad, param_name, cur_loss=None):
        raise NotImplementedError

class Adam(OptimizerBase):
    def __init__(
        self,
        lr=0.001,
        decay1=0.9,
        decay2=0.999,
        eps=1e-7,
        clip_norm=None,
        lr_scheduler=None,
        **kwargs
    ):
        super().__init__(lr, lr_scheduler)

        self.cache = {}
        self.hyperparameters = {
            "id": "Adam",
            "lr": lr,
            "eps": eps,
            "decay1": decay1,
            "decay2": decay2,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        H = self.hyperparameters
        lr, d1, d2 = H["lr"], H["decay1"], H["decay2"]
        eps, cn, sc = H["eps"], H["clip_norm"], H["lr_scheduler"]
        return "Adam(lr={}, decay1={}, decay2={}, eps={}, clip_norm={}, lr_scheduler={})".format(
            lr, d1, d2, eps, cn, sc
        )

    def update(self, param, param_grad, param_name, cur_loss=None):
        """
        Compute the Adam update for a given parameter.

        Parameters
        ----------
        param : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of the parameter to be updated.
        param_grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The gradient of the loss function with respect to `param_name`.
        param_name : str
            The name of the parameter.
        cur_loss : float
            The training or validation loss for the current minibatch. Used for
            learning rate scheduling e.g., by
            :class:`~numpy_ml.neural_nets.schedulers.KingScheduler`. Default is
            None.

        Returns
        -------
        updated_params : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of `param` after applying the Adam update.
        """
        C = self.cache
        H = self.hyperparameters
        d1, d2 = H["decay1"], H["decay2"]
        eps, clip_norm = H["eps"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = {
                "t": 0,
                "mean": np.zeros_like(param_grad),
                "var": np.zeros_like(param_grad),
            }

        # scale gradient to avoid explosion
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        t = C[param_name]["t"] + 1
        var = C[param_name]["var"]
        mean = C[param_name]["mean"]

        # update cache
        C[param_name]["t"] = t
        C[param_name]["var"] = d2 * var + (1 - d2) * param_grad ** 2
        C[param_name]["mean"] = d1 * mean + (1 - d1) * param_grad
        self.cache = C

        # calc unbiased moment estimates and Adam update
        v_hat = C[param_name]["var"] / (1 - d2 ** t)
        m_hat = C[param_name]["mean"] / (1 - d1 ** t)
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        return param - update