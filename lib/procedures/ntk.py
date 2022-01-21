import numpy as np
import torch
from pdb import set_trace as bp


__all__ = ['get_ntk_n']


def get_ntk_n(loader, networks, criterion=torch.nn.CrossEntropyLoss(), train_mode=True, num_batch=-1, num_classes=100):
    for net in networks:
        net.switch_norm('ln')
    device = torch.cuda.current_device()
    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads_x = [[] for _ in range(len(networks))] # size: #training samples. grads of all W from each validation samples
    targets_x_onehot_mean = []; targets_y_onehot_mean = []
    for i, (inputs, targets) in enumerate(loader):
        if num_batch > 0 and i >= num_batch: break
        inputs = inputs.cuda(device=device, non_blocking=True)
        targets = targets.cuda(device=device, non_blocking=True)
        if len(targets.size()) == 1:
            targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
        else:
            assert len(targets.size()) == 2 and targets.size()[1] == num_classes
            targets_onehot = targets.float()
        targets_onehot_mean = targets_onehot - targets_onehot.mean(0)
        targets_x_onehot_mean.append(targets_onehot_mean)
        for net_idx, network in enumerate(networks):
            network = network.cuda()
            network.zero_grad()
            inputs_ = inputs.cuda(device=device, non_blocking=True)
            logit = network.forward_features(inputs_)[1]
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                grads_x[net_idx].append(torch.cat(grad, -1).detach().cpu())
                network.zero_grad()
                torch.cuda.empty_cache()
            network = network.cpu()
            torch.cuda.empty_cache()
    targets_x_onehot_mean = torch.cat(targets_x_onehot_mean, 0)
    conds_x = []
    for idx in range(len(networks)):
        _grads_x = torch.stack([item.cuda() for item in grads_x[idx]], 0)
        grads_x[idx] = [item.cpu() for item in grads_x[idx]]; torch.cuda.empty_cache()
        ntk = torch.einsum('nc,mc->nm', [_grads_x, _grads_x])
        del _grads_x
        try:
            # eigenvalues, _ = torch.symeig(ntk)  # ascending
            eigenvalues, _ = torch.linalg.eigh(ntk, UPLO='L')
        except:
            bp()
            ntk[ntk == float("Inf")] = 0
            ntk[ntk == 0] = ntk.max() # TODO avoid inf in ntk
            eigenvalues, _ = torch.linalg.eigh(ntk + ntk.mean().item() * torch.eye(ntk.shape[0]).cuda() * 1e-4, UPLO='L')  # ascending
        _cond = torch.div(eigenvalues[-1], eigenvalues[0])
        if torch.isnan(_cond):
            conds_x.append(-1) # bad gradients
        else:
            conds_x.append(_cond.item())
    return conds_x
