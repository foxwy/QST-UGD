# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2023-07-10 21:11:25
# @Last Modified by:   yong
# @Last Modified time: 2024-07-30 15:14:45
# @Paper: Efficient factored gradient descent algorithm for quantum state tomography

import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach,
                        _differentiable_doc, _foreach_doc, _maximize_doc)
from typing import List, Optional
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
import copy

__all__ = ["Rprop", "rprop"]


class Rprop(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        etas=(0.5, 1.2),
        step_sizes=(1e-6, 50),
        momentum=0,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < etas[0] < 1.0 < etas[1]:
            raise ValueError("Invalid eta values: {}, {}".format(etas[0], etas[1]))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(
            lr=lr,
            etas=etas,
            step_sizes=step_sizes,
            momentum=momentum,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("momentum", 0)
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)

    def _init_group(self, group, params, funcs, grads, prevs, step_sizes, momentum_buffer_list):
        for p in group["params"]:
            if p.grad is None:
                continue
            params.append(p)
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Rprop does not support sparse gradients")

            grads.append(grad)
            state = self.state[p]

            # State initialization
            if len(state) == 0:
                group["func_prevs"] = funcs
                state["step"] = 0
                state["prev"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if group["momentum"] > 0:
                    state["momentum_buffer"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                if p.dtype.is_complex:
                    # Complex Number should be as if they are two independent real numbers.
                    # Hence the step_size shouldn't be zero for imaginary part.
                    state["step_size"] = (
                        grad.new()
                        .resize_as_(grad)
                        .fill_(complex(group["lr"], group["lr"]))
                    )
                else:
                    state["step_size"] = (
                        grad.new().resize_as_(grad).fill_(group["lr"])
                    )

            if group["momentum"] > 0:
                momentum_buffer_list.append(state["momentum_buffer"])

            prevs.append(state["prev"])
            step_sizes.append(state["step_size"])

            state["step"] += 1

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = []
            funcs = [loss]
            func_prevs = [0]
            grads = []
            prevs = []
            step_sizes = []
            momentum_buffer_list = []
            etaminus, etaplus = group["etas"]
            step_size_min, step_size_max = group["step_sizes"]
            foreach = group["foreach"]
            maximize = group["maximize"]

            self._init_group(group, params, funcs, grads, prevs, step_sizes, momentum_buffer_list)
            func_prevs = group["func_prevs"]

            rprop(
                params,
                funcs,
                func_prevs,
                grads,
                prevs,
                step_sizes,
                momentum_buffer_list,
                step_size_min=step_size_min,
                step_size_max=step_size_max,
                etaminus=etaminus,
                etaplus=etaplus,
                momentum=group["momentum"],
                foreach=foreach,
                maximize=maximize,
                differentiable=group["differentiable"],
            )
            group["func_prevs"] = funcs

        return loss



Rprop.__doc__ = r"""Implements the resilient backpropagation algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \theta_0 \in \mathbf{R}^d \text{ (params)},f(\theta)
                \text{ (objective)},                                                             \\
            &\hspace{13mm}      \eta_{+/-} \text{ (etaplus, etaminus)}, \Gamma_{max/min}
                \text{ (step sizes)}                                                             \\
            &\textbf{initialize} :   g^0_{prev} \leftarrow 0,
                \: \eta_0 \leftarrow \text{lr (learning rate)}                                   \\
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \textbf{for} \text{  } i = 0, 1, \ldots, d-1 \: \mathbf{do}            \\
            &\hspace{10mm}  \textbf{if} \:   g^i_{prev} g^i_t  > 0                               \\
            &\hspace{15mm}  \eta^i_t \leftarrow \mathrm{min}(\eta^i_{t-1} \eta_{+},
                \Gamma_{max})                                                                    \\
            &\hspace{10mm}  \textbf{else if}  \:  g^i_{prev} g^i_t < 0                           \\
            &\hspace{15mm}  \eta^i_t \leftarrow \mathrm{max}(\eta^i_{t-1} \eta_{-},
                \Gamma_{min})                                                                    \\
            &\hspace{15mm}  g^i_t \leftarrow 0                                                   \\
            &\hspace{10mm}  \textbf{else}  \:                                                    \\
            &\hspace{15mm}  \eta^i_t \leftarrow \eta^i_{t-1}                                     \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1}- \eta_t \mathrm{sign}(g_t)             \\
            &\hspace{5mm}g_{prev} \leftarrow  g_t                                                \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to the paper
    `A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm
    <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417>`_.
    """ + r"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        etas (Tuple[float, float], optional): pair of (etaminus, etaplus), that
            are multiplicative increase and decrease factors
            (default: (0.5, 1.2))
        step_sizes (Tuple[float, float], optional): a pair of minimal and
            maximal allowed step sizes (default: (1e-6, 50))
        {foreach}
        {maximize}
        {differentiable}

    """.format(foreach=_foreach_doc, maximize=_maximize_doc, differentiable=_differentiable_doc)

def rprop(
    params: List[Tensor],
    funcs: float,
    func_prevs: float,
    grads: List[Tensor],
    prevs: List[Tensor],
    step_sizes: List[Tensor],
    momentum_buffer_list: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    maximize: bool = False,
    differentiable: bool = False,
    *,
    step_size_min: float,
    step_size_max: float,
    etaminus: float,
    etaplus: float,
    momentum: float,
):
    r"""Functional API that performs rprop algorithm computation.

    See :class:`~torch.optim.Rprop` for details.
    """

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_rprop
    else:
        func = _single_tensor_rprop

    func(
        params,
        funcs,
        func_prevs,
        grads,
        prevs,
        step_sizes,
        momentum_buffer_list,
        step_size_min=step_size_min,
        step_size_max=step_size_max,
        etaminus=etaminus,
        etaplus=etaplus,
        momentum=momentum,
        maximize=maximize,
        differentiable=differentiable,
    )


def _single_tensor_rprop(
    params: List[Tensor],
    funcs: float,
    func_prevs: float,
    grads: List[Tensor],
    prevs: List[Tensor],
    step_sizes: List[Tensor],
    momentum_buffer_list: List[Tensor],
    *,
    step_size_min: float,
    step_size_max: float,
    etaminus: float,
    etaplus: float,
    momentum: float,
    maximize: bool,
    differentiable: bool,
):

    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        prev = prevs[i]
        step_size = step_sizes[i]

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            prev = torch.view_as_real(prev)
            param = torch.view_as_real(param)
            step_size = torch.view_as_real(step_size)
        if differentiable:
            sign = grad.mul(prev.clone()).sign()
        else:
            sign = grad.mul(prev).sign()
        sign[sign.gt(0)] = etaplus
        sign[sign.lt(0)] = etaminus
        sign[sign.eq(0)] = 1

        # update stepsizes with step size updates
        step_size.mul_(sign).clamp_(step_size_min, step_size_max)

        # for dir<0, dfdx=0
        # for dir>=0 dfdx=dfdx
        grad = grad.clone(memory_format=torch.preserve_format)
        if funcs[0] > func_prevs[0]:
            grad[sign.eq(etaminus)] = -grad[sign.eq(etaminus)]
        else:
            grad[sign.eq(etaminus)] = 0

        if momentum > 0:
            buf = momentum_buffer_list[i]
            if is_complex_param:
                buf = torch.view_as_real(buf)
            buf.mul_(momentum).addmul_(grad.sign(), step_size, value=-1)
            param.add_(buf, alpha=1)
        else:
            param.addcmul_(grad.sign(), step_size, value=-1)

        # update parameters
        #param.addcmul_(grad.sign(), step_size, value=-1)
        prev.copy_(grad)


def _multi_tensor_rprop(
    params: List[Tensor],
    funcs: float,
    func_prevs: float,
    grads: List[Tensor],
    prevs: List[Tensor],
    step_sizes: List[Tensor],
    momentum_buffer_list: List[Tensor],
    *,
    step_size_min: float,
    step_size_max: float,
    etaminus: float,
    etaplus: float,
    momentum: float,
    maximize: bool,
    differentiable: bool,
):

    if len(params) == 0:
        return

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = _group_tensors_by_device_and_dtype([params, grads, prevs, step_sizes, momentum_buffer_list])
    for grouped_params, grouped_grads, grouped_prevs, grouped_step_sizes, grouped_momentum_buffer_list in grouped_tensors.values():
        # Handle complex params
        def _view_complex_as_real(tensor_list):
            return [
                torch.view_as_real(t) if torch.is_complex(t) else t for t in tensor_list
            ]

        grouped_grads = _view_complex_as_real(grouped_grads)
        grouped_prevs = _view_complex_as_real(grouped_prevs)
        grouped_params = _view_complex_as_real(grouped_params)
        grouped_step_sizes = _view_complex_as_real(grouped_step_sizes)

        if maximize:
            grouped_grads = torch._foreach_neg(grouped_grads)

        signs = torch._foreach_mul(grouped_grads, grouped_prevs)
        signs = [s.sign() for s in signs]
        for sign in signs:
            sign[sign.gt(0)] = etaplus
            sign[sign.lt(0)] = etaminus
            sign[sign.eq(0)] = 1

        # update stepsizes with step size updates
        torch._foreach_mul_(grouped_step_sizes, signs)
        for step_size in grouped_step_sizes:
            step_size.clamp_(step_size_min, step_size_max)

        # for dir<0, dfdx=0
        # for dir>=0 dfdx=dfdx
        grouped_grads = list(grouped_grads)
        for i in range(len(grouped_grads)):
            grouped_grads[i] = grouped_grads[i].clone(memory_format=torch.preserve_format)
            if funcs[0] > func_prevs[0]:
                grouped_grads[i][signs[i].eq(etaminus)] = - grouped_grads[i][signs[i].eq(etaminus)]
            else:
                grouped_grads[i][signs[i].eq(etaminus)] = 0

        # update parameters
        grad_signs = [grad.sign() for grad in grouped_grads]

        if momentum > 0:
            grouped_momentum_buffer_list = _view_complex_as_real(grouped_momentum_buffer_list)
            torch._foreach_mul_(grouped_momentum_buffer_list, momentum)
            torch._foreach_addcmul_(grouped_momentum_buffer_list, grad_signs, grouped_step_sizes, value=-1)
            torch._foreach_add_(grouped_params, grouped_momentum_buffer_list, alpha=1)
        else:
            torch._foreach_addcmul_(grouped_params, grad_signs, grouped_step_sizes, value=-1)

        for i in range(len(grouped_prevs)):
            grouped_prevs[i].copy_(grouped_grads[i])