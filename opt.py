import numpy as np
import math
import torch
from torch.optim.optimizer import Optimizer, required

import sys
sys.path.append("../")
from batch_svd import batch_svd

class AdaBlockW(Optimizer):
    """
    Stochastic Gradient Methods with Block Diagonal Matrix Adaptation
    followed by Decoupling Weight Decay
    
    For a dense layer, we group each parameter input-neuron wise.
    For a conv layer, we group each parameter filter-wise.
    For a recurrent layer, we group each parameter output-neuron wise.
    
    Args:
        params:         parameters we should optimize
        lr:             learning rate
        betas:          momentum weights
        delta:          dampening parameter for eigendecomposition
        eps:            numerical stability parameter for diagonal updates
        weight_decay:   L2-regularization strength
        groups:         block size of AdaBlock
        gamma:          speed of clipping
        clipping:       clipping or not?
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), delta=1e-4,
                 eps=1e-4, weight_decay=0.0, groups=10):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= delta:
            raise ValueError("Invalid inverse epsilon value: {}".format(inv_eps))
        if not 0.0 <= eps:
            raise ValueError("Invalid diagonal epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 32 >= groups:
            raise ValueError("Batch SVD does not support matrix larger than 32 x 32")
        defaults = dict(lr=lr, betas=betas, delta=delta, eps=eps,
                        weight_decay=weight_decay, groups=groups)
        super(AdaBlockW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaBlockW, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # TODO: randomization, decoupling weight decay
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                dim = len(grad.shape)

                if dim == 1:
                    # bias param or BatchNorm

                    # Perform decoupled weight decay
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                    state = self.state[p]
                     
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1

                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    # TODO: learning rate clipping
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.addcdiv_(-step_size, exp_avg, denom)

                elif dim == 2:
                    # dense weight

                    # Perform decoupled weight decay
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                    state = self.state[p]
                    out_features, in_features = grad.data.shape
                    
                    groups = group['groups']
                    if out_features <= groups:
                        groups = out_features
                    delta = group['delta']

                    batch = int(in_features * out_features / groups)

                    # state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros(batch, groups, groups).cuda()
                        state['I'] = 1e-6 * torch.eye(groups).cuda()

                    exp_avg = state['exp_avg']
                    beta1, beta2 = group['betas']

                    state['step'] += 1

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    grad_transpose = torch.transpose(grad, 0, 1)
                    grad_group = grad_transpose.reshape(-1, groups, 1)
                    
                    # estimate 1st momentum as usual
                    state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad
                    exp_avg_transpose = torch.transpose(state['exp_avg'], 0, 1)
                    exp_avg_group = exp_avg_transpose.reshape(-1, groups, 1)
                    
                    # estimate 2nd momentum matrix (adaptation matrix) based on coordinate partitioning
                    state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * torch.bmm(grad_group, torch.transpose(grad_group, 1, 2))

                    # compute the square root of a matrix
                    Q, S, _ = batch_svd(state['exp_avg_sq'] + state['I'])        # G = Q * S * Q^T
                    eff_S = step_size / (torch.sqrt(S) + delta) 
                    
                    # TODO: spectrum clipping
                    V_inv = torch.bmm(torch.bmm(Q, torch.diag_embed(eff_S, 0, 1)), Q.transpose(1, 2))

                    p_group = torch.transpose(p, 0, 1).reshape(-1, groups, 1)
                    p_group.data = p_group.data - torch.bmm(V_inv, exp_avg_group)

                    p_group = torch.transpose(p_group.reshape(in_features, out_features), 0, 1)
                    p.data = p_group.data

                elif dim == 4:
                    # conv weight
                    out_channels, in_channels, kernel, _ = grad.data.shape
                    if kernel == 1:
                        # Decoupled weight decay
                        p.data.mul_(1 - group['lr'] * group['weight_decay'])

                        # diagonal
                        state = self.state[p]

                        if len(state) == 0:
                            state['step'] = 0
                            state['exp_avg'] = torch.zeros_like(p.data)
                            state['exp_avg_sq'] = torch.zeros_like(p.data)
                        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                        beta1, beta2 = group['betas']

                        state['step'] += 1

                        exp_avg.mul_(beta1).add_(1 - beta1, grad)
                        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                        denom = exp_avg_sq.sqrt().add_(group['eps'])

                        bias_correction1 = 1 - beta1 ** state['step']
                        bias_correction2 = 1 - beta2 ** state['step']
                        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                        p.data.addcdiv_(-step_size, exp_avg, denom)

                    else:
                        # Decoupled weight decay
                        p.data.mul_(1 - group['lr'] * group['weight_decay'])

                        state = self.state[p]
                        batch = out_channels * in_channels

                        delta = group['delta']
                        batch = out_channels * in_channels
                        groups = kernel*kernel

                        # state initialization
                        if len(state) == 0:
                            state['step'] = 0
                            state['exp_avg'] = torch.zeros_like(p.data)
                            state['exp_avg_sq'] = torch.zeros(batch, groups, groups).cuda()
                            state['I'] = 1e-6 * torch.eye(groups).cuda()

                        exp_avg = state['exp_avg']
                        beta1, beta2 = group['betas']

                        state['step'] += 1

                        bias_correction1 = 1 - beta1 ** state['step']
                        bias_correction2 = 1 - beta2 ** state['step']
                        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                        grad_group = grad.reshape(-1, groups, 1)
                        
                        # estimate 1st momentum
                        state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad
                        exp_avg_group = state['exp_avg'].reshape(-1, groups, 1)

                        # estimate 2nd momentum matrix (adaptation matrix)
                        state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * torch.bmm(grad_group, torch.transpose(grad_group, 1, 2))
                        
                        Q, S, _ = batch_svd(state['exp_avg_sq'] + state['I'])
                        eff_S = step_size / (torch.sqrt(S) + delta)
                        
                        # TODO: spectrum clipping
                        V_inv = torch.bmm(torch.bmm(Q, torch.diag_embed(eff_S, 0, 1)), Q.transpose(1, 2))

                        p_group = p.reshape(-1, kernel*kernel, 1)
                        p_group.data = p_group.data - torch.bmm(V_inv, exp_avg_group)
                        
                        p_group = p_group.reshape(out_channels, in_channels, kernel, kernel)
                        p.data = p_group.data
        return loss

class RadaBlockW(Optimizer):
    """
    Stochastic Gradient Methods with Block Diagonal Matrix Adaptation 

    Here, we employ **layer-wise randomized update**
    
    For a dense layer, we group each parameter input-neuron wise.
    For a conv layer, we group each parameter filter-wise.
    For a recurrent layer, we group each parameter output-neuron wise.
    
    Args:
        params:         parameters we should optimize
        lr:             learning rate
        betas:          momentum weights
        delta:          dampening parameter for eigendecomposition
        eps:            numerical stability parameter for diagonal updates
        weight_decay:   strength of weight decay
        groups:         block size of AdaBlock
        gamma:          speed of clipping
        final_lr:       final SGD learning rate
        clipping:       clipping or not?
        k:              the number of random layers with block diagonal approx. at each iteration
        layers:         total layers causing bottleneck
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), 
                 eps=1e-8, weight_decay=0.0, groups=10, 
                 gamma=1e-3, final_lr=0.5, clipping=False, k=required, layers=required):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid diagonal epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 2 <= groups <= 32:
            raise ValueError("Batch SVD does not support matrix larger than 32 x 32")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, groups=groups,
                        gamma=gamma, final_lr=final_lr, clipping=clipping, 
                        k=k, layers=layers)
        super(RadaBlockW, self).__init__(params, defaults)
        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(RadaBlockW, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            total_layers = group['layers']
            k = group['k']

            rand_idx = (torch.randperm(total_layers) + 1)[:k]
            cnt = 0
            for p in group['params']:
                if p.grad is None:
                    print ("Here!")
                    continue

                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad.data
                dim = len(grad.shape)

                if dim == 1:
                    # bias param or BatchNorm

                    state = self.state[p]
                     
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1
                    
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    # learning rate clipping for diagonal case
                    if group['clipping']:
                        final_lr = group['final_lr'] * group['lr'] / base_lr
                        lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                        upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))

                        step_size = torch.full_like(denom, step_size) 
                        step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)
                        p.data.add_(-step_size)
                        
                    else:
                        p.data.addcdiv_(-step_size, exp_avg, denom)

                elif dim == 2:
                    # dense weight
                    cnt += 1 

                    state = self.state[p]
                    out_features, in_features = grad.data.shape
                    
                    groups = group['groups']
                    if out_features <= groups:
                        groups = out_features
                    eps = group['eps']

                    batch = int(in_features * out_features / groups)

                    # state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros(batch, groups, groups).cuda()
                        state['I'] = 1e-6 * torch.eye(groups).cuda()

                    exp_avg = state['exp_avg']
                    beta1, beta2 = group['betas']

                    state['step'] += 1
                    
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    grad_transpose = torch.transpose(grad, 0, 1)
                    grad_group = grad_transpose.reshape(-1, groups, 1)
                    
                    # estimate 1st momentum as usual
                    state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad
                    exp_avg_transpose = torch.transpose(state['exp_avg'], 0, 1)
                    exp_avg_group = exp_avg_transpose.reshape(-1, groups, 1)
                    
                    # estimate 2nd momentum matrix (adaptation matrix) based on coordinate partitioning
                    state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * torch.bmm(grad_group, torch.transpose(grad_group, 1, 2))

                    # block diagonal matrix adaptations
                    if cnt in rand_idx:
                        # compute the square root of a matrix
                        """
                        Q, S, _ = batch_svd(state['exp_avg_sq'] + state['I'])        # G = Q * S * Q^T
                        eff_S = step_size / (torch.sqrt(S) + delta) 
                        """
                        Q, S, _ = batch_svd(state['exp_avg_sq'])
                        S_sqrt_inv = 1 / (S.sqrt())
                        eff_S = 1 / (S_sqrt_inv.add_(1 / eps))
                        eff_S = (step_size / eps) *  (1 - eff_S / eps)

                        # spectrum clipping
                        if group['clipping']:
                            final_lr = group['final_lr'] * group['lr'] / base_lr
                            lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                            upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                            eff_S.clamp_(lower_bound, upper_bound)

                        V_inv = torch.bmm(torch.bmm(Q, torch.diag_embed(eff_S, 0, 1)), Q.transpose(1, 2))

                        p_group = torch.transpose(p, 0, 1).reshape(-1, groups, 1)
                        p_group.data = p_group.data - torch.bmm(V_inv, exp_avg_group)

                        p_group = torch.transpose(p_group.reshape(in_features, out_features), 0, 1)
                        p.data = p_group.data
                    # diagonal approximations
                    else:
                        V_inv = step_size / (torch.diagonal(state['exp_avg_sq'], dim1=-2, dim2=-1).sqrt() + eps)

                        if group['clipping']:
                            final_lr = group['final_lr'] * group['lr'] / base_lr
                            lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                            upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                            V_inv.clamp_(lower_bound, upper_bound)

                        V_inv = torch.diag_embed(V_inv, 0, 1)
                        
                        p_group = torch.transpose(p, 0, 1).reshape(-1, groups, 1)
                        p_group.data = p_group.data - torch.bmm(V_inv, exp_avg_group)

                        p_group = torch.transpose(p_group.reshape(in_features, out_features), 0, 1)
                        p.data = p_group.data

                elif dim == 4:
                    # conv weight
                    out_channels, in_channels, kernel, _ = grad.data.shape
                    if kernel == 1 or kernel > 5:
                        # diagonal
                        state = self.state[p]

                        if len(state) == 0:
                            state['step'] = 0
                            state['exp_avg'] = torch.zeros_like(p.data)
                            state['exp_avg_sq'] = torch.zeros_like(p.data)
                        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                        beta1, beta2 = group['betas']

                        state['step'] += 1
                        
                        exp_avg.mul_(beta1).add_(1 - beta1, grad)
                        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                        denom = exp_avg_sq.sqrt().add_(group['eps'])

                        bias_correction1 = 1 - beta1 ** state['step']
                        bias_correction2 = 1 - beta2 ** state['step']
                        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                        # learning rate clipping for diagonal case
                        if group['clipping']:
                            final_lr = group['final_lr'] * group['lr'] / base_lr
                            lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                            upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))

                            step_size = torch.full_like(denom, step_size) 
                            step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)
                            p.data.add_(-step_size)
                            
                        else:
                            p.data.addcdiv_(-step_size, exp_avg, denom)

                    else:
                        cnt += 1 

                        state = self.state[p]
                        batch = out_channels * in_channels

                        eps = group['eps']
                        batch = out_channels * in_channels
                        groups = kernel*kernel

                        # state initialization
                        if len(state) == 0:
                            state['step'] = 0
                            state['exp_avg'] = torch.zeros_like(p.data)
                            state['exp_avg_sq'] = torch.zeros(batch, groups, groups).cuda()
                            state['I'] = 1e-6 * torch.eye(groups).cuda()

                        exp_avg = state['exp_avg']
                        beta1, beta2 = group['betas']

                        state['step'] += 1
                        
                        bias_correction1 = 1 - beta1 ** state['step']
                        bias_correction2 = 1 - beta2 ** state['step']
                        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                        grad_group = grad.reshape(-1, groups, 1)
                        
                        # estimate 1st momentum
                        state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad
                        exp_avg_group = state['exp_avg'].reshape(-1, groups, 1)

                        # estimate 2nd momentum matrix (adaptation matrix)
                        state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * torch.bmm(grad_group, torch.transpose(grad_group, 1, 2))
                        
                        # block diagonal matrix adaptations
                        if cnt in rand_idx:
                            # compute the square root of a matrix
                            """
                            Q, S, _ = batch_svd(state['exp_avg_sq'] + state['I'])        # G = Q * S * Q^T
                            eff_S = step_size / (torch.sqrt(S) + delta) 
                            """
                            Q, S, _ = batch_svd(state['exp_avg_sq'])
                            S_sqrt_inv = 1 / (S.sqrt())
                            eff_S = 1 / (S_sqrt_inv.add_(1 / eps))
                            eff_S = (step_size / eps) * (1 - eff_S / eps)

                            # spectrum clipping
                            if group['clipping']:
                                final_lr = group['final_lr'] * group['lr'] / base_lr
                                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                                eff_S.clamp_(lower_bound, upper_bound)

                            V_inv = torch.bmm(torch.bmm(Q, torch.diag_embed(eff_S, 0, 1)), Q.transpose(1, 2))

                            p_group = p.reshape(-1, kernel*kernel, 1)
                            p_group.data = p_group.data - torch.bmm(V_inv, exp_avg_group)

                            p_group = p_group.reshape(out_channels, in_channels, kernel, kernel)
                            p.data = p_group.data
                        # diagonal approximations
                        else:
                            V_inv = step_size / (torch.diagonal(state['exp_avg_sq'], dim1=-2, dim2=-1).sqrt() + eps)

                            if group['clipping']:
                                final_lr = group['final_lr'] * group['lr'] / base_lr
                                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                                V_inv.clamp_(lower_bound, upper_bound)

                            V_inv = torch.diag_embed(V_inv, 0, 1)
                            
                            p_group = p.reshape(-1, kernel*kernel, 1)
                            p_group.data = p_group.data - torch.bmm(V_inv, exp_avg_group)

                            p_group = p_group.reshape(out_channels, in_channels, kernel, kernel)
                            p.data = p_group.data
        return loss

class AdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

class SGDW(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                d_p = p.grad.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

